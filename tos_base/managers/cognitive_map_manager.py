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
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import copy

from ..core.room import Room, BaseRoom
from ..core.object import Object, Agent, Gate
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
)

# Utils
from ..utils.cogmap.transforms import (
    transform_baseroom,
    br_from_anchor_to_initial,
)
from ..utils.cogmap.metrics import compute_map_metrics
from ..utils.cogmap.consistency import (
    local_vs_global_consistency,
    rooms_vs_global_consistency,
    map_vs_relations_consistency,
    relations_consistency,
)
from ..utils.cogmap.types import BaseCogMetrics, MapCogMetrics, RelationMetrics, ConsistencySummary
from ..utils.cogmap.analysis import (
    compute_error_aggregates,
    compute_correctness_aggregates,
    compute_consistency_aggregates,
    calculate_cogmap_per_turn,
    compute_evaluation_correctness_aggregates,
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
    connectivity: Dict[str, float] = field(default_factory=dict)
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)
    gt_room_state_full: Optional['BaseRoom'] = None
    gt_json_full: Dict[str, Any] = field(default_factory=dict)
    metrics_full: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "connectivity": self.connectivity,
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
            "gt_room_state_full": self.gt_room_state_full.to_dict() if self.gt_room_state_full else {},
            "gt_json_full": self.gt_json_full,
            "metrics_full": (self.metrics_full.to_dict() if self.metrics_full.valid else {}),
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
class RoomsCogMapTurnLog(BaseCogMapTurnLog):
    pred_rooms_state: Dict[str, 'BaseRoom'] = field(default_factory=dict)
    gt_rooms_state: Dict[str, 'BaseRoom'] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "pred_rooms_state": {k: v.to_dict() for k, v in self.pred_rooms_state.items()} if self.pred_rooms_state else {},
            "gt_rooms_state": {k: v.to_dict() for k, v in self.gt_rooms_state.items()} if self.gt_rooms_state else {},
        })
        return out


@dataclass
class RelationsCogMapTurnLog(BaseCogMapTurnLog):
    pred_relations: Dict[str, str] = field(default_factory=dict)
    gt_relations: Dict[str, str] = field(default_factory=dict)
    gt_relations_full: Dict[str, str] = field(default_factory=dict)
    metrics_full: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "pred_relations": self.pred_relations,
            "gt_relations": self.gt_relations,
            "gt_relations_full": self.gt_relations_full,
            "metrics_full": (self.metrics_full.to_dict() if self.metrics_full.valid else {}),
        })
        return out


@dataclass
class CognitiveMapTurnLog:
    """Aggregate per-type logs for one turn."""
    global_log: Optional[GlobalCogMapTurnLog] = None
    local_log: Optional[LocalCogMapTurnLog] = None
    rooms_log: Optional[RoomsCogMapTurnLog] = None
    relations_log: Optional[RelationsCogMapTurnLog] = None
    consistency: Optional[ConsistencySummary] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.global_log:
            out["global"] = self.global_log.to_dict()
        if self.local_log:
            out["local"] = self.local_log.to_dict()
        if self.rooms_log:
            out["rooms"] = self.rooms_log.to_dict()
        if self.relations_log:
            out["relations"] = self.relations_log.to_dict()
        if self.consistency:
            out["consistency"] = self.consistency.to_dict()
        return out


class CognitiveMapManager:
    """Evaluate cognitive map JSON against ground truth."""
    
    DEFAULT_COGMAP_SUMMARY = {
        "global": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
        "gates": {"conn_acc": 0.0},
        "relations": {"dir": 0.0, "dist": 0.0, "overall": 0.0},
        "extraction_success_rate": 0.0,
        "n_successful": 0,
        "n_evaluations": 0,
    }
    
    def __init__(self, cogmap_type: str = "standard", pos_allow_scale: bool = False, scope: str = "all"):
        """Initialize cognitive map manager."""
        self.explore_logs: List[CognitiveMapTurnLog] = []
        self.evaluate_log: Optional[CognitiveMapTurnLog] = None
        self.cogmap_summary = copy.deepcopy(self.DEFAULT_COGMAP_SUMMARY)

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
        self._prev_room_id: int | None = None

    def get_supported_types(self) -> List[str]:
        return ["global", "local", "rooms", "relations"]

    def evaluate_cogmap_type(self, assistant_response: str, gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], map_type: str) -> Optional[BaseCogMapTurnLog]:
        """Extract JSON and evaluate a single cogmap type (global|local|rooms|relations). Only compute what's needed for the given type."""
        self._register_active_entry_gate(gt_room)
        t = (map_type or "global").lower()
        json_dict = self._extract_json_from_text(assistant_response)
        if json_dict is None or gt_room is None:
            return BaseCogMapTurnLog(type=t, extraction_success=False, original_response=assistant_response, metrics=BaseCogMetrics.invalid())
        all_item_names = {o.name for o in gt_room.all_objects}
        observed_set: set[str] = set(all_item_names if observed_items is None else [str(x) for x in observed_items])
        visible_names = self._visible_object_names(gt_room, gt_agent)

        if t == "global":
            jd = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent)
            pred_global_br = self._parse_global(jd)
            pred_gates = self._parse_gates(jd)
            pred_global_br = pred_global_br or BaseRoom(objects=[], name="pred_global")
            gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
            full_global = transform_baseroom(self._baseroom_from_gt(gt_room, gt_agent), gt_agent.init_pos, gt_agent.init_ori)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_global(pred_global_br, pred_gates or {}, gt_global_br, full_global, assistant_response, gt_room)

        if t == "local":
            jd = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent)
            pred_local_br = self._parse_local(jd)
            pred_local_br = pred_local_br or BaseRoom(objects=[], name="pred_local")
            gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_local(pred_local_br, gt_local_br, assistant_response, gt_room, gt_agent)

        if t == "rooms":
            jd = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent)
            pred_rooms_map = self._parse_rooms(jd)
            pred_rooms_map = pred_rooms_map or {}
            gt_rooms_map = self._build_gt_room_baserooms(gt_room, gt_agent, observed_set)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_rooms(pred_rooms_map, gt_rooms_map, assistant_response)

        if t == "relations":
            # No map preprocessing needed; relations are a flat dict of pairs
            pred_relations = self._parse_predicted_relations(json_dict)
            full_global = transform_baseroom(self._baseroom_from_gt(gt_room, gt_agent), gt_agent.init_pos, gt_agent.init_ori)
            return self._eval_relations(pred_relations, full_global, observed_set, assistant_response)

        raise ValueError(f"Invalid map type: {t}")

    def _eval_global(self, pred_global_br: BaseRoom, pred_gates: Dict[str, Any], gt_global_br: BaseRoom, gt_room_state_full: BaseRoom, assistant_response: str, gt_room: Room) -> GlobalCogMapTurnLog:
        metrics = self._compare_baserooms(pred_global_br, gt_global_br)
        pred_json = self.baseroom_to_json(pred_global_br, include_gates=True)
        connectivity = {"conn_acc": float(self._evaluate_gate_connections(pred_gates, gt_room))}
        gt_json = self.baseroom_to_json(gt_global_br, include_gates=True)
        gt_json_full = self.baseroom_to_json(gt_room_state_full, include_gates=True)
        metrics_full = self._compare_baserooms(pred_global_br, gt_room_state_full)
        return GlobalCogMapTurnLog(
            type="global",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_global_br,
            metrics=metrics,
            connectivity=connectivity,
            gt_room_state=gt_global_br,
            gt_json=gt_json,
            gt_room_state_full=gt_room_state_full,
            gt_json_full=gt_json_full,
            metrics_full=metrics_full,
        )

    def _eval_local(self, pred_local_br: BaseRoom, gt_local_br: BaseRoom, assistant_response: str, gt_room: Room, gt_agent: Agent) -> LocalCogMapTurnLog:
        gt_local = gt_local_br
        if len(gt_local.objects) > 0:
            gt_local = self._build_gt_local_with_gates(gt_room, gt_agent)
        metrics = self._compare_baserooms(pred_local_br, gt_local if len(gt_local.objects) > 0 else gt_local_br)
        pred_json = self.baseroom_to_json(pred_local_br, include_gates=True)
        return LocalCogMapTurnLog(
            type="local",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_local_br,
            metrics=metrics,
            gt_room_state=(gt_local if len(gt_local.objects) > 0 else gt_local_br),
            gt_json=(self.baseroom_to_json(gt_local, include_gates=True) if len(gt_local.objects) > 0 else self.baseroom_to_json(gt_local_br, include_gates=True)),
        )

    def _eval_rooms(self, pred_rooms_map: Dict[str, BaseRoom], gt_rooms_map: Dict[int, BaseRoom], assistant_response: str) -> RoomsCogMapTurnLog:
        per_room: List[MapCogMetrics] = []
        for rid in sorted(gt_rooms_map.keys()):
            gt_br = gt_rooms_map[rid]
            if len(gt_br.objects) == 0:
                continue
            pred_br = pred_rooms_map.get(str(rid)) or pred_rooms_map.get(rid) or BaseRoom(objects=[], name=f"pred_room_{rid}")
            per_room.append(self._compare_baserooms(pred_br, gt_br))
        metrics = MapCogMetrics.average(per_room)
        pred_json = {rid: self.baseroom_to_json(pred_rooms_map.get(str(rid)) or BaseRoom(objects=[], name=f"pred_room_{rid}"), include_gates=False) for rid in gt_rooms_map.keys()}
        pred_rooms_state = {str(rid): br for rid, br in pred_rooms_map.items()}
        gt_rooms_state = {str(rid): br for rid, br in gt_rooms_map.items()}
        return RoomsCogMapTurnLog(
            type="rooms",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=None,
            metrics=metrics,
            pred_rooms_state=pred_rooms_state,
            gt_rooms_state=gt_rooms_state,
        )

    # =============================== Relations helpers/eval ===============================
    def _parse_predicted_relations(self, json_data: Dict[str, Any]) -> Dict[str, str]:
        from ..utils.relation_codes import decode_relation_codes, make_ordered_pair_key, parse_pair_key
        out: Dict[str, str] = {}
        assert isinstance(json_data, dict), f"json_data must be a dict, but got {type(json_data)}"

        candidates = []
        if isinstance(json_data, dict):
            candidates.append(json_data)
        candidates.append(json_data)
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            for k, v in cand.items():
                if not isinstance(k, str):
                    continue
                a, b = parse_pair_key(k)
                if not a or not b:
                    continue
                key = make_ordered_pair_key(a, b)
                if isinstance(v, str):
                    d, r = decode_relation_codes(v)
                    if d and r:
                        out[key] = f"({d}, {r})"
                elif isinstance(v, dict):
                    d, r = str(v.get('dir', '')).strip().lower(), str(v.get('dist', '')).strip().lower()
                    if d and r:
                        d1, r1 = decode_relation_codes(f"({d},{r})")
                        if d1 and r1:
                            out[key] = f"({d1}, {r1})"
        return out

    # Removed; use room_to_ordered_relations directly

    @staticmethod
    def _relations_accuracies(pred: Dict[str, str], gt: Dict[str, str]) -> Tuple[float, float, float]:
        from ..utils.relation_codes import decode_relation_codes
        if not gt:
            return 0.0, 0.0, 0.0
        tot = len(gt)
        dir_correct = dist_correct = both_correct = 0
        for pair_key, gt_value in gt.items():
            gt_dir_code, gt_dist_code = decode_relation_codes(gt_value)
            pred_dir_code = pred_dist_code = ""
            pval = pred.get(pair_key)
            if isinstance(pval, str):
                pred_dir_code, pred_dist_code = decode_relation_codes(pval)
            if pred_dir_code == gt_dir_code:
                dir_correct += 1
            if pred_dist_code == gt_dist_code:
                dist_correct += 1
            if pred_dir_code == gt_dir_code and pred_dist_code == gt_dist_code:
                both_correct += 1
        return dir_correct / tot, dist_correct / tot, both_correct / tot

    def _eval_relations(self, pred_relations: Dict[str, str], gt_room_state_full: BaseRoom, observed_set: set[str], assistant_response: str) -> RelationsCogMapTurnLog:
        # Observed: include only observed names; include initial_pos at agent.init_pos; exclude agent
        from ..utils.relationship_utils import room_to_ordered_relations
        # Try to find agent initial pos from any Agent present in the full room state
        agent_obj = next((o for o in gt_room_state_full.objects if isinstance(o, Agent)), None)
        agent_init_pos = agent_obj.init_pos

        gt_relations_obs = room_to_ordered_relations(
            gt_room_state_full,
            include_names=set(observed_set),
            include_initial_pos=True,
            agent_init_pos=agent_init_pos,
        )
        # Full: all names; include initial_pos at agent.init_pos; exclude agent
        all_names = {o.name for o in gt_room_state_full.objects if o.name != 'agent'}
        gt_relations_full = room_to_ordered_relations(
            gt_room_state_full,
            include_names=all_names,
            include_initial_pos=True,
            agent_init_pos=agent_init_pos,
        )
        dir_acc_obs, dist_acc_obs, overall_obs = self._relations_accuracies(pred_relations, gt_relations_obs)
        dir_acc_full, dist_acc_full, overall_full = self._relations_accuracies(pred_relations, gt_relations_full)
        return RelationsCogMapTurnLog(
            type="relations",
            extraction_success=True,
            original_response=assistant_response,
            pred_json={},
            pred_room_state=None,
            metrics=RelationMetrics(dir=float(dir_acc_obs), dist=float(dist_acc_obs), overall=float(overall_obs), valid=True),
            pred_relations=pred_relations,
            gt_relations=gt_relations_obs,
            gt_relations_full=gt_relations_full,
            metrics_full=RelationMetrics(dir=float(dir_acc_full), dist=float(dist_acc_full), overall=float(overall_full), valid=True),
        )

    def evaluate_cogmaps(self, responses_by_type: Dict[str, str], gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], mode: str | None = None) -> CognitiveMapTurnLog:
        """Evaluate multiple types and record one aggregate log for the turn."""
        out = CognitiveMapTurnLog()
        for map_type_key, resp in (responses_by_type or {}).items():
            if not isinstance(resp, str):
                continue
            single = self.evaluate_cogmap_type(resp, gt_room, gt_agent, observed_items, map_type_key)
            if single is None:
                continue
            setattr(out, f"{single.type}_log", single)
        # Consistency fields per turn
        summary = ConsistencySummary()
        if out.local_log and out.global_log and out.local_log.extraction_success and out.global_log.extraction_success:
            cm = local_vs_global_consistency(
                out.local_log.pred_room_state,
                out.global_log.pred_room_state,
                gt_agent,
                allow_scale=bool(self.config.get('pos_allow_scale', False)),
                pos_norm_L=self._pos_norm_L,
            )
            summary.local_vs_global = cm
        # Rooms vs Global (only when both predicted)
        if out.rooms_log and out.global_log and out.rooms_log.extraction_success and out.global_log.extraction_success:
            avg, per_room = rooms_vs_global_consistency(
                out.rooms_log.pred_rooms_state or {},
                out.global_log.pred_room_state,
                gt_room,
                gt_agent,
                self.entry_gate_by_room,
                allow_scale=bool(self.config.get('pos_allow_scale', False)),
                pos_norm_L=self._pos_norm_L,
            )
            summary.rooms_vs_global_avg = avg
            summary.rooms_vs_global_per_room = per_room
        # Map vs Relations consistency
        if out.global_log and out.relations_log and out.relations_log.extraction_success and out.global_log.extraction_success:
            score = map_vs_relations_consistency(
                out.relations_log.pred_relations or {},
                out.global_log.pred_room_state,
            )
            summary.map_vs_relations = float(score)
        # Relations self-consistency
        if out.relations_log and out.relations_log.extraction_success:
            score_rel = relations_consistency(out.relations_log.pred_relations or {})
            summary.relations_consistency = float(score_rel)
        out.consistency = summary
        return out
            

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict], exp_type: str = None) -> Dict[str, Any]:
        """Aggregate cognitive map metrics per scenario.

        exp_type in {
            'active': error + consistency + correctness,
            'passive': correctness (global only),
        }
        """
        assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"

        # Always compute these once; then select portions
        correctness = compute_correctness_aggregates(env_data_list)
        
        if exp_type == 'active':
            error = compute_error_aggregates(env_data_list)
            consistency = compute_consistency_aggregates(env_data_list)
            per_turn_update = calculate_cogmap_per_turn(env_data_list, mode='update')
            per_turn_full = calculate_cogmap_per_turn(env_data_list, mode='full')
            return {
                'exploration': {
                    'error': error,
                    'correctness': {
                        'last_global_vs_gt_full': correctness.get('last_global_vs_gt_full', {}),
                    },
                    'consistency': consistency
                },
                'evaluation': {
                    'correctness': {
                        'global_full': compute_evaluation_correctness_aggregates(env_data_list)
                    },
                },
                'cogmap_update_per_turn': per_turn_update,
                'cogmap_full_per_turn': per_turn_full,
            }
        elif exp_type == 'passive':
            return {
                'exploration': {
                    'correctness': {
                        'global_full': correctness.get('last_global_vs_gt_full', {})
                    }
                }
            }

        raise ValueError(f"Invalid scenario: {exp_type}")
    
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
            if not isinstance(position, list) or len(position) != 2:
                continue
            pos = np.array([float(position[0]), float(position[1])])
            facing = obj_info.get('facing', None)
            if isinstance(facing, str):
                ori = direction_mapping.get(facing.lower(), direction_mapping['north'])
                has_orientation = True
            else:
                ori = np.array([0, 0])
                has_orientation = False
            objects.append(Object(name=str(obj_name), pos=pos, ori=ori, has_orientation=has_orientation))
        if len(objects) == 0:
            return None
        return BaseRoom(objects=objects, name=room_name)

    def _parse_global(self, json_data: Dict[str, Any]) -> Optional[BaseRoom]:
        global_sec = json_data.get('global') if any(k in json_data for k in ("global", "local", "rooms", "gates")) else json_data
        return self._parse_section_to_baseroom(global_sec, "pred_global") if isinstance(global_sec, dict) else None

    def _parse_local(self, json_data: Dict[str, Any]) -> Optional[BaseRoom]:
        local_sec = json_data.get('local') if isinstance(json_data, dict) else None
        return self._parse_section_to_baseroom(local_sec, "pred_local") if isinstance(local_sec, dict) else None

    def _parse_rooms(self, rooms_sec: Dict[str, Any]) -> Dict[str, BaseRoom]:
        rooms_map: Dict[str, BaseRoom] = {}
        if isinstance(rooms_sec, dict):
            for rid, sec in rooms_sec.items():
                if isinstance(sec, dict):
                    br = self._parse_section_to_baseroom(sec, f"pred_room_{rid}")
                    if br is not None:
                        rooms_map[str(rid)] = br
        return rooms_map

    def _parse_gates(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        gates_sec = json_data.get('gates') if isinstance(json_data, dict) else None
        return gates_sec if isinstance(gates_sec, dict) else {}

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

    def _build_gt_local_with_gates(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        for g in gt_room.gates:
            objs.append(Object(name=g.name, pos=g.pos.copy(), ori=g.ori.copy(), has_orientation=True))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return transform_baseroom(raw, gt_agent.pos, gt_agent.ori)
    
    def _build_gt_room_baserooms(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> Dict[int, BaseRoom]:
        out: Dict[int, BaseRoom] = {}
        if not isinstance(gt_room, Room):
            return out
        for rid in sorted(gt_room.objects_by_room.keys()):
            gate_name = self.entry_gate_by_room.get(int(rid))
            if gate_name is None: # no entry gate for this room
                continue
            gate = next((g for g in gt_room.gates if g.name == gate_name), None)
            anchor_pos, anchor_ori = gate.pos, gate.get_ori_for_room(int(rid))
            # exclude origin gate and agent; include room objects only
            objs: List[Object] = []
            for name in gt_room.objects_by_room[int(rid)]:
                if name == gate.name or name not in observed_set:
                    continue
                o = gt_room.get_object_by_name(name)
                objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=o.has_orientation))
            out[int(rid)] = transform_baseroom(BaseRoom(objects=objs, name=f'gt_room_{rid}'), anchor_pos, anchor_ori)
        return out
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
            allow_scale=bool(self.config.get('pos_allow_scale', True)),
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
        from ..actions.base import BaseAction
        names = set()
        for o in gt_room.objects:
            if BaseAction._is_visible(gt_agent, o):
                names.add(o.name)
        return names

    def _preprocess_predicted(self, json_data: Dict[str, Any], observed: set[str], visible: set[str], gt_room: Room, gt_agent: Agent) -> Dict[str, Any]:
        jd = copy.deepcopy(json_data) if isinstance(json_data, dict) else {}
        gate_names = {g.name for g in gt_room.gates}

        def _norm_face_global(f):
            """For global section - directions should already be absolute cardinals"""
            if not isinstance(f, str):
                return f
            return f.strip().lower()
        
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

        def _strip_conf_and_faces_global(obj_map: Dict[str, Any]) -> Dict[str, Any]:
            out = {}
            for name, info in (obj_map or {}).items():
                if not isinstance(info, dict):
                    continue
                # drop confidence
                new_info = {k: v for k, v in info.items() if k != "confidence" and k != "origin"}
                # normalize facing
                if "facing" in new_info:
                    new_info["facing"] = _norm_face_global(new_info["facing"])
                out[name] = new_info
            return out
        
        def _strip_conf_and_faces_local(obj_map: Dict[str, Any], anchor_ori) -> Dict[str, Any]:
            out = {}
            for name, info in (obj_map or {}).items():
                if not isinstance(info, dict):
                    continue
                # drop confidence
                new_info = {k: v for k, v in info.items() if k != "confidence" and k != "origin"}
                # normalize facing
                if "facing" in new_info:
                    new_info["facing"] = _norm_face_local(new_info["facing"], anchor_ori)
                out[name] = new_info
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

        # --- Global: keep observed + gates + agent; drop initial_pos ---
        if isinstance(jd.get("global"), dict):
            g = jd["global"]
            keep = set(observed) | gate_names | {"agent"}
            global_dict = {}
            for k, v in g.items():
                should_keep, preferred_key = _should_keep_key(k, keep)
                if should_keep:
                    global_dict[preferred_key] = v
            jd["global"] = _strip_conf_and_faces_global(global_dict)

        # --- Local: drop origin + keep only visible objects ---
        if isinstance(jd.get("local"), dict):
            loc = jd["local"]
            agent_ori = gt_agent.ori
            if "objects" in loc:
                local_dict = {}
                for k, v in loc["objects"].items():
                    should_keep, preferred_key = _should_keep_key(k, visible)
                    if should_keep:
                        local_dict[preferred_key] = v
                jd["local"] = _strip_conf_and_faces_local(local_dict, agent_ori)
            else:
                local_dict = {}
                for k, v in loc.items():
                    should_keep, preferred_key = _should_keep_key(k, visible)
                    if should_keep:
                        local_dict[preferred_key] = v
                jd["local"] = _strip_conf_and_faces_local(local_dict, agent_ori)

        # --- Rooms: drop origin + keep only observed objects ---
        rooms = jd.get("rooms") if isinstance(jd, dict) else None
        if isinstance(rooms, dict):
            out_rooms = {}
            for rid, sec in rooms.items():
                # get rid of "origin" if present
                if not isinstance(sec, dict):
                    continue
                inner = sec.get("objects", sec)  # sometimes wrapped in {"origin":..., "objects":{...}}
                keep = {
                    n
                    for n in observed
                    if n in gt_room.room_by_object
                    and gt_room.room_by_object[n] == int(rid)
                }
                # Get gate orientation for this room
                gate_name = self.entry_gate_by_room.get(int(rid))
                if gate_name:
                    gate = next((g for g in gt_room.gates if g.name == gate_name), None)
                    if gate:
                        gate_ori = gate.get_ori_for_room(int(rid))
                        room_dict = {}
                        for k, v in inner.items():
                            should_keep, preferred_key = _should_keep_key(k, keep)
                            if should_keep:
                                room_dict[preferred_key] = v
                        out_rooms[str(rid)] = _strip_conf_and_faces_local(room_dict, gate_ori)
                    else:
                        # fallback if gate not found
                        room_dict = {}
                        for k, v in inner.items():
                            should_keep, preferred_key = _should_keep_key(k, keep)
                            if should_keep:
                                room_dict[preferred_key] = v
                        out_rooms[str(rid)] = _strip_conf_and_faces_global(room_dict)
                else:
                    # fallback if no entry gate
                    room_dict = {}
                    for k, v in inner.items():
                        should_keep, preferred_key = _should_keep_key(k, keep)
                        if should_keep:
                            room_dict[preferred_key] = v
                    out_rooms[str(rid)] = _strip_conf_and_faces_global(room_dict)
            jd["rooms"] = out_rooms
        return jd


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

    # =============================== Gates evaluation =============================== 
    def _gt_gate_connections_dict(self, gt_room: Room) -> Dict[str, Any]:
        """Return {gate_name: {'connects':[room_id_a, room_id_b]}} from GT room."""
        out: Dict[str, Any] = {}
        for g in gt_room.gates:
            # expect Gate.room_id like [a,b]
            if isinstance(g.room_id, (list, tuple)) and len(g.room_id) == 2:
                out[g.name] = {"connects": [int(g.room_id[0]), int(g.room_id[1])]}
        return out

    def _evaluate_gate_connections(self, pred_gates: Dict[str, Any], gt_room: Room) -> float:
        if not isinstance(gt_room, Room):
            return 0.0
        gt_gates = self._gt_gate_connections_dict(gt_room)
        if not gt_gates:
            return 0.0
        correct = tot = 0
        for gate_name, gt_info in gt_gates.items():
            gt_conn = sorted([int(x) for x in gt_info.get("connects", [])])
            pred = pred_gates.get(gate_name, {}) if isinstance(pred_gates, dict) else {}
            pred_conn = pred.get("connects", []) if isinstance(pred, dict) else []
            try:
                pred_conn_int = sorted([int(x) for x in pred_conn])
            except Exception:
                pred_conn_int = []
            if gt_conn == pred_conn_int:
                correct += 1
            tot += 1

        return float(correct) / float(tot) if tot > 0 else 0.0


if __name__ == "__main__":
    pass