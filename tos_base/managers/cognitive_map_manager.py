"""
Cognitive Map Manager

Minimal, modular evaluator for cognitive maps.

Responsibilities:
- Extract JSON from LLM response
- Transform JSON sections (global/local/rooms/gates) into BaseRoom-compatible data
- Evaluate global, local, room maps (dir/facing/pos) using consistent coordinates
- Evaluate gates connectivity
- Log all results per turn for summary aggregation
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import copy
from ..actions.base import BaseAction
from ..core.room import Room, BaseRoom
from ..core.object import Object, Agent, Gate
# Utils
from ..utils.cogmap.transforms import (
    transform_baseroom,
)
from ..utils.cogmap.metrics import compute_map_metrics
from ..utils.cogmap.consistency import (
    local_vs_global_consistency,
    stability,
)
from ..utils.cogmap.types import BaseCogMetrics, MapCogMetrics, ConsistencySummary, UnexploredMetrics
from ..utils.cogmap.analysis import (
    get_last_exploration_cogmap,
    avg_nested_dicts,
)
from ..utils.cogmap.candidates import calculate_other_candidates_metrics
from ..utils.cogmap.unexplored import (
    evaluate_unexplored_predictions,
    parse_unexplored_response,
    aggregate_unexplored_metrics,
)


@dataclass
class BaseCogMapTurnLog:
    """Common fields for all cogmap types."""
    type: str
    extraction_success: bool = False
    original_response: str = ""
    pred_json: Dict[str, Any] = field(default_factory=dict)
    pred_room_state: Optional['BaseRoom'] = None
    metrics: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "extraction_success": self.extraction_success,
            "original_response": self.original_response,
            "pred_json": self.pred_json,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {},
            "metrics": (self.metrics.to_dict() if self.metrics.valid else {}),
        }

@dataclass
class GlobalCogMapTurnLog(BaseCogMapTurnLog):
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)
    gt_room_state_full: Optional['BaseRoom'] = None
    gt_json_full: Dict[str, Any] = field(default_factory=dict)
    metrics_full: BaseCogMetrics = field(default_factory=BaseCogMetrics)
    metric_agent: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
            "gt_room_state_full": self.gt_room_state_full.to_dict() if self.gt_room_state_full else {},
            "gt_json_full": self.gt_json_full,
            "metrics_full": (self.metrics_full.to_dict() if self.metrics_full.valid else {}),
            "metric_agent": (self.metric_agent.to_dict() if self.metric_agent.valid else {}),
        })
        return out

@dataclass
class LocalCogMapTurnLog(BaseCogMapTurnLog):
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
        })
        return out

@dataclass
class UnexploredCogMapTurnLog(BaseCogMapTurnLog):
    """Turn log for unexplored area predictions."""
    all_candidate_points: List[Tuple[int, int]] = field(default_factory=list)
    pred_points: List[Tuple[int, int]] = field(default_factory=list)
    correct_points: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "all_candidate_points": [[int(x), int(y)] for x, y in (self.all_candidate_points or [])],
            "pred_points": [[int(x), int(y)] for x, y in (self.pred_points or [])],
            "correct_points": [[int(x), int(y)] for x, y in (self.correct_points or [])],
        })
        return out


@dataclass
class CognitiveMapTurnLog:
    """Aggregate per-type logs for one turn."""
    global_log: Optional[GlobalCogMapTurnLog] = None
    local_log: Optional[LocalCogMapTurnLog] = None
    unexplored_log: Optional[UnexploredCogMapTurnLog] = None
    consistency: Optional[ConsistencySummary] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.global_log:
            out["global"] = self.global_log.to_dict()
        if self.local_log:
            out["local"] = self.local_log.to_dict()
        if self.unexplored_log:
            out["unexplored"] = self.unexplored_log.to_dict()
        if self.consistency:
            out["consistency"] = self.consistency.to_dict()
        return out


class CognitiveMapManager:
    """Evaluate cognitive map JSON against ground truth."""    
    def __init__(self, cogmap_type: str = "standard", pos_allow_scale: bool = False, scope: str = "all"):
        """Initialize cognitive map manager."""
        self.config = {
            "cogmap_type": cogmap_type,
            "pos_allow_scale": bool(pos_allow_scale),
            "scope": (scope if scope in ("global", "all") else "all"),
        }
        # room_id -> first-entry gate name
        self.entry_gate_by_room: dict[int, str] = {}
        # position normalization scale (computed once in global frame)
        self._pos_norm_L: float | None = None
        self._start_room_id: int | None = None

    def evaluate_cogmap_type(self, assistant_response: str, gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], map_type: str) -> Optional[BaseCogMapTurnLog]:
        """Extract JSON and evaluate a single cogmap type (global|local|rooms). Only compute what's needed for the given type."""
        self._register_active_entry_gate(gt_room)
        t = (map_type or "global").lower()
        json_dict = self._extract_json_from_text(assistant_response)
        if json_dict is None or gt_room is None:
            m = MapCogMetrics.invalid()
            return BaseCogMapTurnLog(type=t, extraction_success=False, original_response=assistant_response, metrics=m)
        all_item_names = {o.name for o in gt_room.all_objects}
        observed_set: set[str] = set(all_item_names if observed_items is None else [str(x).replace('_', ' ') for x in observed_items])
        visible_names = self._visible_object_names(gt_room, gt_agent)

        if t == "global":
            pred_global_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
            full_global = transform_baseroom(self._baseroom_from_gt(gt_room, gt_agent), gt_agent.init_pos, gt_agent.init_ori)
            agent_br = self._build_gt_global_agent_baseroom(gt_room, gt_agent)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_global(pred_global_br, gt_global_br, full_global, agent_br, assistant_response, json_dict)
        
        if t == "local":
            pred_local_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
            # If nothing is visible, skip this local turn (mark invalid so aggregations ignore it).
            if not gt_local_br.objects:
                return LocalCogMapTurnLog(
                    type="local",
                    extraction_success=True,
                    original_response=assistant_response,
                    pred_json=json_dict,
                    pred_room_state=pred_local_br,
                    metrics=MapCogMetrics.invalid(),
                    gt_room_state=gt_local_br,
                    gt_json={},
                )
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_local(pred_local_br, gt_local_br, assistant_response, json_dict)

        raise ValueError(f"Invalid map type: {t}")

    def evaluate_unexplored(
        self,
        assistant_response: str,
        all_candidate_coords: Optional[List[Tuple[int, int]]],
        correct_coords: List[Tuple[int, int]],
    ) -> UnexploredCogMapTurnLog:
        """Evaluate unexplored area predictions using coordinate selection.
        
        The LLM is presented with candidate coordinates and must select those
        corresponding to unexplored regions.
        
        Args:
            assistant_response: LLM response text containing unexplored predictions
            correct_coords: List of correct unexplored (x, y) coordinates
            
        Returns:
            UnexploredCogMapTurnLog with evaluation metrics
        """
        assert correct_coords and all_candidate_coords, "No correct or candidate coordinates provided"
        # Parse predicted coordinates
        pred_coords = parse_unexplored_response(assistant_response)
        # if not all_candidate_coords or not correct_coords:
        #     return UnexploredCogMapTurnLog(
        #         type="unexplored",
        #         extraction_success=True,
        #         original_response=assistant_response,
        #         pred_json={"parsed_from_text": True, "predicted_coords": [[int(x), int(y)] for x, y in pred_coords]},
        #         all_candidate_points=all_candidate_coords,
        #         pred_points=pred_coords,
        #         correct_points=correct_coords,
        #         metrics=UnexploredMetrics(overall=1.0, precision=1.0, recall=1.0, valid=True),
        #     )

        # Evaluate predictions
        metrics = evaluate_unexplored_predictions(pred_coords, correct_coords)
        
        return UnexploredCogMapTurnLog(
            type="unexplored",
            extraction_success=True,
            original_response=assistant_response,
            pred_json={"parsed_from_text": True, "predicted_coords": [[int(x), int(y)] for x, y in pred_coords]},
            all_candidate_points=all_candidate_coords,
            pred_points=pred_coords,
            correct_points=correct_coords,
            metrics=metrics,
        )

    def _eval_global(self, pred_global_br: BaseRoom,  gt_global_br: BaseRoom, gt_room_state_full: BaseRoom, agent_br: BaseRoom, assistant_response: str, pred_json: Dict) -> GlobalCogMapTurnLog:
        gt_json = self.baseroom_to_json(gt_global_br, include_gates=True)
        metrics = self._compare_baserooms(pred_global_br, gt_global_br)
        gt_json_full = self.baseroom_to_json(gt_room_state_full, include_gates=True)
        metrics_full = self._compare_baserooms(pred_global_br, gt_room_state_full)
        metric_agent = self._compare_baserooms(pred_global_br, agent_br)
        return GlobalCogMapTurnLog(
            type="global",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_global_br,
            metrics=metrics,
            gt_room_state=gt_global_br,
            gt_json=gt_json,
            gt_room_state_full=gt_room_state_full,
            gt_json_full=gt_json_full,
            metrics_full=metrics_full,
            metric_agent=metric_agent,
        )

    def _eval_local(self, pred_local_br: BaseRoom, gt_local_br: BaseRoom, assistant_response: str, pred_json: Dict) -> LocalCogMapTurnLog:
        metrics = self._compare_baserooms(pred_local_br, gt_local_br)
        return LocalCogMapTurnLog(
            type="local",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_local_br,
            metrics=metrics,
            gt_room_state=gt_local_br,
            gt_json=self.baseroom_to_json(gt_local_br, include_gates=True),
        )
    
    def evaluate_cogmaps(
        self,
        responses_by_type: Dict[str, str],
        gt_room: Room,
        gt_agent: Agent,
        observed_items: Optional[List[str]],
        all_correct_coords: Optional[List[Tuple[int, int]]] = None,
        all_candidate_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> CognitiveMapTurnLog:
        """Evaluate multiple types and record one aggregate log for the turn.
        
        Args:
            responses_by_type: Dict mapping map_type to LLM response
            gt_room: Ground truth room
            gt_agent: Ground truth agent
            observed_items: List of observed item names
            all_correct_coords: List of correct unexplored (x, y) coordinates
        """
        out = CognitiveMapTurnLog()
        for map_type_key, resp in (responses_by_type or {}).items():
            if not isinstance(resp, str):
                continue
            if map_type_key == "unexplored":
                single = self.evaluate_unexplored(resp, all_candidate_coords, all_correct_coords or [])
            else:
                single = self.evaluate_cogmap_type(resp, gt_room, gt_agent, observed_items, map_type_key)
            setattr(out, f"{single.type}_log", single)
        # Consistency fields per turn
        summary = ConsistencySummary()
        if (
            out.local_log and out.global_log
            and out.local_log.extraction_success and out.global_log.extraction_success
            and out.local_log.metrics.valid and out.global_log.metrics.valid
        ):
            cm = local_vs_global_consistency(
                out.local_log.pred_room_state,
                out.global_log.pred_room_state,
                gt_agent,
                allow_scale=bool(self.config.get('pos_allow_scale', False)),
                pos_norm_L=self._pos_norm_L,
            )
            summary.local_vs_global = cm
        
        out.consistency = summary
        return out
            

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict], exp_type: str = None) -> Dict[str, Any]:
        """Aggregate cognitive map metrics per scenario.

        exp_type in {
            'active': error + consistency + correctness,
            'passive': correctness (global only),
        }
        Prefer precomputed per-sample metrics when available.
        """
        assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"

        pre_list = [cogmap for s in env_data_list if (cogmap := (s.get('metrics') or {}).get('cogmap')) is not None]
        if exp_type == 'active':
            exploration = avg_nested_dicts([m.get('exploration') or {} for m in pre_list])
            evaluation = avg_nested_dicts([m.get('evaluation') or {} for m in pre_list])

            per_turn_list = [(m.get('per_turn_metrics') or {}) for m in pre_list if isinstance(m, dict)]
            update_turn = avg_nested_dicts([{'cogmap_update_per_turn': d.get('cogmap_update_per_turn') or {}} for d in per_turn_list]).get('cogmap_update_per_turn', {})
            full_turn = avg_nested_dicts([{'cogmap_full_per_turn': d.get('cogmap_full_per_turn') or {}} for d in per_turn_list]).get('cogmap_full_per_turn', {})
            self_tracking_turn = avg_nested_dicts([{'self_tracking_per_turn': d.get('self_tracking_per_turn') or {}} for d in per_turn_list]).get('self_tracking_per_turn', {})
            other_candidates_f1_turn = avg_nested_dicts([{'other_candidates_f1_per_turn': d.get('other_candidates_f1_per_turn') or []} for d in per_turn_list]).get('other_candidates_f1_per_turn', [])
            other_candidates_count_turn = avg_nested_dicts([{'other_candidates_count_per_turn': d.get('other_candidates_count_per_turn') or []} for d in per_turn_list]).get('other_candidates_count_per_turn', [])
            unexplored_f1_turn = avg_nested_dicts([{'unexplored_f1_per_turn': d.get('unexplored_f1_per_turn') or []} for d in per_turn_list]).get('unexplored_f1_per_turn', [])

            per_turn_metrics = {
                'cogmap_update_per_turn': update_turn,
                'cogmap_full_per_turn': full_turn,
                'self_tracking_per_turn': self_tracking_turn,
                'other_candidates_f1_per_turn': other_candidates_f1_turn,
                'other_candidates_count_per_turn': other_candidates_count_turn,
                'unexplored_f1_per_turn': unexplored_f1_turn,
            }
            return {
                'exploration': exploration,
                'evaluation': evaluation if evaluation else {'correctness': {}},
                'per_turn_metrics': per_turn_metrics,
            }
        if exp_type == 'passive':
            exploration = avg_nested_dicts([m.get('exploration') or {} for m in pre_list])
            return {'exploration': {'correctness': {'global_full': (exploration.get('correctness') or {}).get('global_full', {})}}}

        # Default: average nested
        return avg_nested_dicts(pre_list)

    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any], exp_type: str | None = None) -> Dict[str, Any]:
        """Aggregate cognitive-map metrics within a single sample (over turns).
        Returns exploration error/correctness/consistency and per-turn global metrics.
        """
        # Helper: get exploration turns' cogmap logs
        turn_logs = env_data.get('env_turn_logs') or []
        cog_logs = []
        exp_logs = []
        for t in turn_logs:
            if t.get('is_exploration_phase', False) and t.get('cogmap_log'):
                cog_logs.append(t['cogmap_log'])
                exp_logs.append(t.get('exploration_log') or {})
        if not cog_logs:
            return {}
        # Use shared helper to find last exploration cogmap
        
        last = get_last_exploration_cogmap(env_data)
        # Average metrics over turns
        def _avg_maps(dicts: List[Dict[str, Any]], path: List[str]) -> MapCogMetrics:
            mats: List[MapCogMetrics] = []
            for d in dicts:
                cur = d
                ok = True
                for key in path:
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, dict):
                    m = MapCogMetrics.from_dict(cur)
                    if m.valid:
                        mats.append(m)
            return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()

        def _d(m: MapCogMetrics) -> Dict[str, float]:
            return m.to_dict() if m.valid else {}

        error = {
            'local_vs_gt_local_avg': _d(_avg_maps(cog_logs, ['local', 'metrics'])),
            'global_vs_gt_global_avg': _d(_avg_maps(cog_logs, ['global', 'metrics'])),
            'agent_vs_gt_agent_avg': _d(_avg_maps(cog_logs, ['global', 'metric_agent'])),
        }

        # Correctness: last global_full
        correctness = {
            'last_global_vs_gt_full': (lambda _m: (_m.to_dict() if _m.valid else {}))(MapCogMetrics.from_dict((((last or {}).get('global') or {}).get('metrics_full') or {}))),
        }

        # Consistency
        # local_vs_global average over turns
        def _avg_consistency_lvsg(dicts: List[Dict[str, Any]]) -> MapCogMetrics:
            mats: List[MapCogMetrics] = []
            for d in dicts:
                cm = (d.get('consistency') or {}).get('local_vs_global') or {}
                m = MapCogMetrics.from_dict(cm)
                if m.valid:
                    mats.append(m)
            return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()

        # Compute stability metrics (now returns two values: update and stability_check)
        update_metrics, stability_check_metrics = stability(env_data)

        consistency = {
            'local_vs_global_avg': _d(_avg_consistency_lvsg(cog_logs)),
            'update_avg': float(np.mean(update_metrics)) if update_metrics else None,
            'stability_avg': _d(MapCogMetrics.average(stability_check_metrics)),
        }

        # Per-turn global metrics (list)
        per_turn_update, per_turn_full, per_turn_self_tracking = CognitiveMapManager.compute_per_turn_global_metrics(cog_logs)
        other_f1, other_count = calculate_other_candidates_metrics(env_data)
        # Unexplored: keep per-turn F1 for plotting, but aggregate by points (not turn-average).
        unexp_f1_per_turn, unexp_f1_avg, unexp_distance_hit_corr = aggregate_unexplored_metrics(cog_logs, exp_logs)

        def _avg_list(vals: List[Optional[float]]) -> float | None:
            xs = [float(v) for v in (vals or []) if isinstance(v, (int, float))]
            return float(np.mean(xs)) if xs else None

        other_f1_avg = _avg_list(other_f1)
        other_count_avg = _avg_list(other_count)

        if exp_type == 'passive':
            return {
                'exploration': {
                    'correctness': {
                        'global_full': correctness['last_global_vs_gt_full']
                    }
                }
            }

        per_turn_metrics = {
            'cogmap_update_per_turn': per_turn_update,
            'cogmap_full_per_turn': per_turn_full,
            'self_tracking_per_turn': per_turn_self_tracking,
            'other_candidates_f1_per_turn': other_f1,
            'other_candidates_count_per_turn': other_count,
            'unexplored_f1_per_turn': unexp_f1_per_turn,
        }

        return {
            'exploration': {
                'error': error,
                'correctness': correctness,
                'consistency': consistency,
                'other_candidates': {
                    'f1_avg': other_f1_avg,
                    'count_score_avg': other_count_avg,
                },
                'unexplored': {
                    'f1_avg': unexp_f1_avg,
                    'distance_hit_corr': unexp_distance_hit_corr,
                },
            },
            'per_turn_metrics': per_turn_metrics,
        }

    @staticmethod
    def compute_per_turn_global_metrics(cog_logs: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """Return (update, full, self_tracking) per-turn global metric lists."""
        per_turn_update = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        per_turn_full = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        per_turn_self_tracking = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        for d in cog_logs:
            g = d.get('global') or {}
            mu = MapCogMetrics.from_dict(g.get('metrics') or {})
            mf = MapCogMetrics.from_dict(g.get('metrics_full') or {})
            ma = MapCogMetrics.from_dict(g.get('metric_agent') or {})
            per_turn_update['dir'].append(float(mu.dir) if mu.valid else None)
            per_turn_update['facing'].append(float(mu.facing) if mu.valid else None)
            per_turn_update['pos'].append(float(mu.pos) if mu.valid else None)
            per_turn_update['overall'].append(float(mu.overall) if mu.valid else None)
            per_turn_full['dir'].append(float(mf.dir) if mf.valid else None)
            per_turn_full['facing'].append(float(mf.facing) if mf.valid else None)
            per_turn_full['pos'].append(float(mf.pos) if mf.valid else None)
            per_turn_full['overall'].append(float(mf.overall) if mf.valid else None)
            per_turn_self_tracking['dir'].append(float(ma.dir) if ma.valid else None)
            per_turn_self_tracking['facing'].append(float(ma.facing) if ma.valid else None)
            per_turn_self_tracking['pos'].append(float(ma.pos) if ma.valid else None)
            per_turn_self_tracking['overall'].append(float(ma.overall) if ma.valid else None)
        return per_turn_update, per_turn_full, per_turn_self_tracking
    
    # register entry gates for active exploratoin
    def _register_active_entry_gate(self, gt_room) -> None:
        """
        Register entry gates for rooms based on room structure.
        For simplicity, assign the first gate connecting to room 1 as the entry gate for each room.
        Room 1 is considered the starting room and doesn't get an entry gate.
        """
        if not hasattr(gt_room, 'gates') or not gt_room.gates:
            return
            
        # Set room 1 as the starting room
        if self._start_room_id is None:
            self._start_room_id = 1

        # For each room (except room 1), find its connection to room 1 or already processed rooms
        processed_rooms = {self._start_room_id}
        
        # Keep processing until no more rooms can be processed
        changed = True
        while changed:
            changed = False
            for g in gt_room.gates:
                if len(g.room_id) == 2:
                    room_a, room_b = int(g.room_id[0]), int(g.room_id[1])
                    
                    # If one room is processed and the other isn't, register the gate for the unprocessed room
                    if room_a in processed_rooms and room_b not in processed_rooms:
                        if room_b not in self.entry_gate_by_room:
                            self.entry_gate_by_room[room_b] = g.name
                            processed_rooms.add(room_b)
                            changed = True
                    elif room_b in processed_rooms and room_a not in processed_rooms:
                        if room_a not in self.entry_gate_by_room:
                            self.entry_gate_by_room[room_a] = g.name
                            processed_rooms.add(room_a)
                            changed = True

    # =============================== Parsing helpers =============================== 
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON content from text."""
        # Try fenced blocks first
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = fenced if fenced else []

        # Fallback: scan for outermost balanced braces
        stack, start = [], None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i+1])
                    start = None

        # Try to load the largest candidate
        candidates.sort(key=len, reverse=True)
        for cand in candidates:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
        return None
    
    def _parse_section_to_baseroom(self, mapping: Dict[str, Any], room_name: str) -> Optional[BaseRoom]:
        """Parse a single section (object_name -> attrs) to BaseRoom.
        Keeps 'agent' as a regular object for evaluation symmetry.
        """
        direction_mapping = {
            "north": np.array([0, 1]),
            "south": np.array([0, -1]),
            "east": np.array([1, 0]),
            "west": np.array([-1, 0])
        }
        objects: List[Object] = []
        for obj_name, obj_info in mapping.items():
            if not isinstance(obj_info, dict):
                continue
            position = obj_info.get('position')
            if not isinstance(position, list) or len(position) != 2 or not all(isinstance(x, (int, float, str)) for x in position):
                continue
            pos = np.array([float(position[0]), float(position[1])])
            facing = obj_info.get('facing', None)
            if isinstance(facing, str):
                ori = direction_mapping.get(facing.lower(), direction_mapping['north'])
                has_orientation = True
            else:
                ori = np.array([0, 0])
                has_orientation = False
            objects.append(Object(name=str(obj_name).replace('_', ' '), pos=pos, ori=ori, has_orientation=has_orientation))

        return BaseRoom(objects=objects, name=room_name)


    # =============================== GT constructors =============================== 

    def _baseroom_from_gt(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        objs: List[Object] = []
        # include all non-gate objects
        for o in gt_room.objects:
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=o.has_orientation))
        # include gates
        for g in gt_room.gates:
            objs.append(Object(name=g.name, pos=g.pos.copy(), ori=g.ori.copy(), has_orientation=True))
        # include agent
        objs.append(Agent(name='agent', pos=gt_agent.pos.copy(), ori=gt_agent.ori.copy(), has_orientation=True))
        return BaseRoom(objects=objs, name='gt')

    def _build_gt_global_agent_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        return self._filter_br_by_names(br, {"agent"})
    
    def _build_gt_global_baseroom(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = set(observed_set) | {"agent"}
        return self._filter_br_by_names(br, keep)

    def _build_gt_local_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return transform_baseroom(raw, gt_agent.pos, gt_agent.ori)

    def baseroom_to_json(self, room: BaseRoom, include_gates: bool = True) -> Dict[str, Any]:
        """
        Convert a BaseRoom into a cognitive map–style JSON.

        Args:
            room (BaseRoom): the BaseRoom instance to convert
            include_gates (bool): whether to include gates in the output

        Returns:
            Dict[str, Any]: JSON-like dictionary following the cognitive map schema
        """
        ori_mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
        out: Dict[str, Any]={}
        # Objects (includes agent if present)
        for obj in room.objects:
            facing = ori_mapping.get(tuple(obj.ori), "")
            out[obj.name] = {
                "position": [int(obj.pos[0]), int(obj.pos[1])],
                "facing": facing
            }

        # Gates
        if include_gates and room.gates:
            for g in room.gates:
                gate_facing = ori_mapping.get(tuple(g.ori), "")
                out[g.name] = {
                    "position": [int(g.pos[0]), int(g.pos[1])],
                    "facing": gate_facing
                }

        return out

    # =============================== Room comparisons =============================== 

    def _compare_baserooms(self, pred_room: BaseRoom, gt_room: BaseRoom) -> MapCogMetrics:
        m = compute_map_metrics(
            pred_room,
            gt_room,
            allow_scale=bool(self.config.get('pos_allow_scale', False)),
            pos_norm_L=self._pos_norm_L,
        )
        return m

    # =============================== Filters and preprocessing =============================== 
    def _filter_br_by_names(self, br: Optional[BaseRoom], keep: set[str]) -> BaseRoom:
        if br is None:
            return BaseRoom(objects=[], name='empty')
        objs = [o for o in br.objects if o.name in keep]
        return BaseRoom(objects=objs, name=br.name)

    def _visible_object_names(self, gt_room: Room, gt_agent: Agent) -> set[str]:
        names = set()
        for o in gt_room.all_objects:
            if BaseAction._is_visible(gt_agent, o):
                names.add(o.name)
        return names

    def _preprocess_predicted(self, json_data: Dict[str, Any], observed: set[str], visible: set[str], gt_room: Room, gt_agent: Agent, map_type: str) -> Dict[str, Any]:
        jd = copy.deepcopy(json_data) if isinstance(json_data, dict) else {}
        gate_names = {g.name for g in gt_room.gates}
        
        def _norm_face_local(f, anchor_ori):
            """For local/room sections - convert relative directions to absolute based on anchor orientation"""
            if not isinstance(f, str):
                return f
            s = f.strip().lower()
            # anchor_ori is like [0,1] for north, [1,0] for east, etc.
            if tuple(anchor_ori) == (0, 1):  # north
                mapping = {"+x": "east", "-x": "west", "+y": "north", "-y": "south"}
            elif tuple(anchor_ori) == (1, 0):  # east
                mapping = {"+x": "south", "-x": "north", "+y": "east", "-y": "west"}
            elif tuple(anchor_ori) == (0, -1):  # south
                mapping = {"+x": "west", "-x": "east", "+y": "south", "-y": "north"}
            elif tuple(anchor_ori) == (-1, 0):  # west
                mapping = {"+x": "north", "-x": "south", "+y": "west", "-y": "east"}
            else:
                # fallback to identity
                return s
            return mapping.get(s, s)

        def _norm_face_global(f):
            """Best-effort: normalize common variants (incl. ego terms) to cardinal directions."""
            if not isinstance(f, str):
                return f
            s = f.strip().lower()
            mapping = {
                # canonical
                "north": "north", "n": "north",
                "south": "south", "s": "south",
                "east": "east", "e": "east",
                "west": "west", "w": "west",
                # local axis variants (treat as global frame where north=+y, east=+x)
                "+y": "north", "-y": "south",
                "+x": "east", "-x": "west",
                # ego variants (robustness; assume global frame uses initial-facing-as-north)
                "forward": "north", "front": "north", "ahead": "north",
                "back": "south", "backward": "south", "behind": "south",
                "right": "east",
                "left": "west",
            }
            return mapping.get(s, s)

        def _norm_map(obj_map: Dict[str, Any], keep: set = None, anchor_ori = None, face_fn=None) -> Dict[str, Any]:
            out = {}
            for name, info in (obj_map or {}).items():
                if not isinstance(info, dict):
                    continue
                # Apply keep filter if provided
                if keep is not None:
                    should_keep, preferred_key = _should_keep_key(name, keep)
                    if not should_keep:
                        continue
                    name = preferred_key
                # normalize facing
                if "facing" in info:
                    if face_fn is not None:
                        info["facing"] = face_fn(info["facing"])
                    elif anchor_ori is not None:
                        info["facing"] = _norm_face_local(info["facing"], anchor_ori)
                out[name] = info
            return out

        def _should_keep_key(key: str, keep_set: set) -> tuple[bool, str]:
            """Check if key should be kept and return the preferred key name from keep_set"""
            if key in keep_set:
                return True, key
            # Check if key with underscores matches any keep element without underscores
            if '_' in key:
                key_no_underscore = key.replace('_', '')
                for keep_item in keep_set:
                    if keep_item.replace('_', '') == key_no_underscore:
                        return True, keep_item

            return False, key

        def _flatten_nested_json(jd: Dict[str, Any]) -> Dict[str, Any]:
            """Convert nested JSON format (objects/gates arrays) to flat format."""
            if not (isinstance(jd, dict) and ("objects" in jd or "gates" in jd)):
                return jd
                
            flat = {}

            # keep other top-level dict entries (e.g., "agent")
            for k, v in jd.items():
                if k in ("objects", "gates"):
                    continue
                if isinstance(v, dict):
                    flat[str(k).replace('_', ' ')] = v

            def _norm_name(x):
                s = x.get("label") or x.get("name") or x.get("id") or x.get("type") or ""
                s = str(s).strip()
                return s.replace("_", " ") if s else ""

            def _emit(name, info):
                if not name:
                    return
                # require a position-like field
                pos = info.get("position") or info.get("pos") or info.get("xy")
                if pos is None:
                    return
                out = {"position": pos}
                if "facing" in info:
                    out["facing"] = info["facing"]
                # keep other fields, but don't clobber position/facing
                for k, v in info.items():
                    if k not in ("position", "pos", "xy", "facing", "id", "label", "name", "type"):
                        out[k] = v
                flat[name] = out

            def _add(sec):
                if isinstance(sec, list):
                    for it in sec:
                        if isinstance(it, dict):
                            _emit(_norm_name(it), it)
                elif isinstance(sec, dict):
                    for name, info in sec.items():
                        if isinstance(info, dict):
                            _emit(str(name).replace('_', ' '), info)

            _add(jd.get("objects"))
            _add(jd.get("gates"))
            return flat or jd

        # --- Global: keep observed + gates + agent; also handle list-based sections ---
        if map_type == "global":
            # Flatten {"objects":[...], "gates":[...]} into {name: {position, facing, ...}}
            jd = _flatten_nested_json(jd)
            keep = set(observed) | gate_names | {"agent"}
            jd = _norm_map(jd, keep, face_fn=_norm_face_global)
            return self._parse_section_to_baseroom(jd, "pred_global") or BaseRoom(objects=[], name="pred_global")

        # --- Local: drop origin + keep only visible objects ---
        if map_type == "local":
            # Handle nested format if present
            jd = _flatten_nested_json(jd)
            if "objects" in jd:
                jd = jd["objects"]
            jd = _norm_map(jd, visible, gt_agent.ori)
            return self._parse_section_to_baseroom(jd, "pred_local") or BaseRoom(objects=[], name="pred_local")

        raise ValueError(f"Invalid map_type: {map_type}")


    def _ensure_pos_norm_L(self, gt_room: Room, gt_agent: Agent) -> None:
        if self._pos_norm_L is not None:
            return
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = {o.name for o in gt_room.objects}
        br = self._filter_br_by_names(br, keep)
        if not br.objects:
            self._pos_norm_L = 1.0
            return
        P = np.array([o.pos for o in br.objects], dtype=float)
        L = float(np.sqrt((P ** 2).sum(axis=1).mean()))
        self._pos_norm_L = (L if L > 0 else 1.0)


def test_evaluate_cogmaps():
    """Test function to demonstrate calling CognitiveMapManager.evaluate_cogmaps method."""
    import json
    import numpy as np

    # Path to the JSON file
    json_file_path = "results-test/GLM-4.5V/5fde50e6fe43edcb/vision/active/think/exploration_turn_logs.json"

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    turn_log = data[2]

    print(f"Selected turn number: {turn_log.get('turn_number', 'Unknown')}")
    print(f"Total turns available: {len(data)}")

    observed_items = turn_log.get('observed_items', [])
    room_state_data = turn_log.get('room_state', {})
    agent_state_data = turn_log.get('agent_state', {})

    # Prepare responses by type
    # Since cogmap_response is None, let's use the original_response from cogmap_log
    responses_by_type = {}
    cogmap_log = turn_log.get('cogmap_log', {})

    # Extract original responses from each cogmap type'global', 'local', 
    for map_type in ['unexplored']:
        if map_type in cogmap_log and isinstance(cogmap_log[map_type], dict):
            original_response = cogmap_log[map_type].get('original_response', '')
            if original_response:
                responses_by_type[map_type] = original_response


    print(f"\n📋 Constructing gt_room and gt_agent from turn data...")

    # Import required classes
    from ..core.room import Room
    from ..core.object import Agent

    # Construct gt_room directly from room_state_data using Room.from_dict
    gt_room = Room.from_dict(room_state_data)

    # Construct gt_agent from agent_state_data using Agent.from_dict
    gt_agent = Agent.from_dict(agent_state_data)

    print(f"✅ Constructed gt_room with {len(gt_room.objects)} objects and {len(gt_room.gates)} gates")
    print(f"✅ Constructed gt_agent at position {gt_agent.pos} facing {gt_agent.ori}")

    # Create CognitiveMapManager instance
    manager = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")

    print(f"\n🚀 Calling manager.evaluate_cogmaps()...")

    # Extract correct coordinates directly
    all_correct_coords_raw = turn_log['exploration_log'].get('all_correct_coords', [])
    all_correct_coords = [(int(pt[0]), int(pt[1])) for pt in all_correct_coords_raw] if all_correct_coords_raw else []

    # Call the actual evaluate_cogmaps method
    result = manager.evaluate_cogmaps(
        responses_by_type, 
        gt_room, 
        gt_agent, 
        observed_items, 
        all_correct_coords=all_correct_coords
    )

    print(f"✅ Successfully called evaluate_cogmaps!")
    print(f"📊 Result type: {type(result)}")

    # Display the results
    if result:
        print(f"\n📊 New evaluation results:")
        result_dict = result.to_dict()
        for map_type, log_data in result_dict.items():
            if isinstance(log_data, dict) and 'metrics' in log_data:
                metrics = log_data['metrics']
                print(f"   {map_type}: {metrics}")


if __name__ == "__main__":
    # Run the test function
    test_evaluate_cogmaps()