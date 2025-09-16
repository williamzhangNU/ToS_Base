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




@dataclass
class CogMapMetrics:
    """Container for similarity metrics with helpers."""
    dir: float = 0.0
    facing: float = 0.0
    pos: float = 0.0
    overall: float = 0.0
    valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        return {"dir": self.dir, "facing": self.facing, "pos": self.pos, "overall": self.overall}

    @staticmethod
    def average(items: List['CogMapMetrics']) -> 'CogMapMetrics':
        valid_items = [i for i in items if isinstance(i, CogMapMetrics) and i.valid]
        if not valid_items:
            return CogMapMetrics.invalid()
        return CogMapMetrics(
            dir=float(np.mean([i.dir for i in valid_items])),
            facing=float(np.mean([i.facing for i in valid_items])),
            pos=float(np.mean([i.pos for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )

    @classmethod
    def invalid(cls) -> 'CogMapMetrics':
        return cls(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=False)


@dataclass
class BaseCogMapTurnLog:
    """Common fields for all cogmap types."""
    type: str
    extraction_success: bool = False
    original_response: str = ""
    pred_json: Dict[str, Any] = field(default_factory=dict)
    pred_room_state: Optional['BaseRoom'] = None
    metrics: CogMapMetrics = field(default_factory=CogMapMetrics)

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
    metrics_full: CogMapMetrics = field(default_factory=CogMapMetrics)

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
class CognitiveMapTurnLog:
    """Aggregate per-type logs for one turn."""
    global_log: Optional[GlobalCogMapTurnLog] = None
    local_log: Optional[LocalCogMapTurnLog] = None
    rooms_log: Optional[RoomsCogMapTurnLog] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.global_log:
            out["global"] = self.global_log.to_dict()
        if self.local_log:
            out["local"] = self.local_log.to_dict()
        if self.rooms_log:
            out["rooms"] = self.rooms_log.to_dict()
        return out


# =============================== transforms =============================== 

def _rotation_matrix_from_ori(ori: np.ndarray) -> np.ndarray:
    """Rotate world into anchor frame so that +Y aligns with anchor forward.
    Mappings chosen so anchor_ori -> [0,1] (north).
    """
    ori_to_R = {
        (0, 1): np.array([[1, 0], [0, 1]]),           # north → identity
        (1, 0): np.array([[0, -1], [1, 0]]),           # east  → +90°
        (0, -1): np.array([[-1, 0], [0, -1]]),         # south → 180°
        (-1, 0): np.array([[0, 1], [-1, 0]]),          # west  → -90°
    }
    key = tuple(int(x) for x in (ori.tolist() if hasattr(ori, 'tolist') else ori))
    return ori_to_R.get(key, ori_to_R[(0, 1)])


def _transform_point(pos_world: np.ndarray, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = _rotation_matrix_from_ori(anchor_ori)
    return (R @ (pos_world.astype(float) - anchor_pos.astype(float))).astype(float)


def _transform_ori(ori_world: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = _rotation_matrix_from_ori(anchor_ori)
    v = (R @ ori_world.astype(float)).astype(int)
    vx, vy = int(np.sign(v[0])), int(np.sign(v[1]))
    return np.array([vx, vy], dtype=int)


def _transform_baseroom(room: BaseRoom, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> BaseRoom:
    objects: List[Object] = []
    for obj in room.objects:
        p = _transform_point(obj.pos, anchor_pos, anchor_ori)
        o = obj.ori
        # if obj.has_orientation:
        #     o = _transform_ori(obj.ori, anchor_ori)
        objects.append(Object(name=obj.name, pos=p, ori=o, has_orientation=obj.has_orientation))
    return BaseRoom(objects=objects, name=room.name)

def _inv_transform_point(pos_local: np.ndarray, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    """Local->world: world = R^T @ local + anchor_pos"""
    R = _rotation_matrix_from_ori(anchor_ori)
    return (R.T @ pos_local.astype(float)) + anchor_pos.astype(float)

def _inv_transform_ori(ori_local: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    """Local->world orientation."""
    R = _rotation_matrix_from_ori(anchor_ori)
    v = (R.T @ ori_local.astype(float))
    return np.array([int(np.sign(v[0])), int(np.sign(v[1]))], dtype=int)

def _br_from_anchor_to_initial(br_anchor: BaseRoom, anchor_pos: np.ndarray, anchor_ori: np.ndarray, gt_agent: Agent) -> BaseRoom:
    """Take a room expressed in an anchor frame and return it in the initial/global frame."""
    # 1) anchor frame -> world
    objs_world = []
    for o in br_anchor.objects:
        p_w = _inv_transform_point(o.pos, anchor_pos, anchor_ori)
        if o.has_orientation:
            ori_w = _inv_transform_ori(o.ori, anchor_ori)
        else:
            ori_w = o.ori
        objs_world.append(Object(name=o.name, pos=p_w, ori=ori_w, has_orientation=o.has_orientation))
    br_world = BaseRoom(objects=objs_world, name=br_anchor.name)

    # 2) world -> initial 
    return _transform_baseroom(
        br_world,
        anchor_pos=np.array(gt_agent.init_pos, dtype=float),
        anchor_ori=np.array(gt_agent.init_ori, dtype=int),
    )

class CognitiveMapManager:
    """Evaluate cognitive map JSON against ground truth."""
    
    DEFAULT_COGMAP_SUMMARY = {
        "global": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
        "gates": {"conn_acc": 0.0},
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
        return ["global", "local", "rooms"]
        
    def evaluate_cogmap_type(self, assistant_response: str, gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], map_type: str, mode: str) -> Optional[BaseCogMapTurnLog]:
        """Extract JSON and evaluate a single cogmap type (global|local|rooms)."""
        assert mode in ("explore", "evaluate"), f"Invalid mode: {mode}"
        self._register_active_entry_gate(gt_room)
        json_dict = self._extract_json_from_text(assistant_response)
        if json_dict is None or gt_room is None:
            return None
        all_item_names = {o.name for o in gt_room.all_objects}
        observed_set: set[str] = set(all_item_names if observed_items is None else [str(x) for x in observed_items])
        visible_names = self._visible_object_names(gt_room, gt_agent)
        json_dict = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent)
        pred_global_br, pred_local_br, pred_rooms_map, pred_gates = self._parse_predicted_maps(json_dict)
        pred_global_br = pred_global_br or BaseRoom(objects=[], name="pred_global")
        pred_local_br = pred_local_br or BaseRoom(objects=[], name="pred_local")
        pred_rooms_map = pred_rooms_map or {}
        pred_gates = pred_gates or {}
        gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
        gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
        gt_rooms_map = self._build_gt_room_baserooms(gt_room, gt_agent, observed_set)
        self._ensure_pos_norm_L(gt_room, gt_agent)

        map_type_lower = (map_type or "global").lower()
        metrics = CogMapMetrics.invalid()
        pred_json: Dict[str, Any] = {}
        pred_state: Optional[BaseRoom] = None
        gt_room_state: Optional[BaseRoom] = None
        if map_type_lower == "global":
            # Global includes ALL gates in both pred and GT; and provide full vs observed GT
            metrics = self._compare_baserooms(pred_global_br, gt_global_br)
            pred_state = pred_global_br
            pred_json = self.baseroom_to_json(pred_global_br, include_gates=True)
            gt_room_state = gt_global_br
        elif map_type_lower == "local":
            if len(gt_local_br.objects) > 0:
                # include gates in local GT
                gt_local_with_gates = self._build_gt_local_with_gates(gt_room, gt_agent)
                metrics = self._compare_baserooms(pred_local_br, gt_local_with_gates)
                gt_local_br = gt_local_with_gates
            pred_state = pred_local_br
            pred_json = self.baseroom_to_json(pred_local_br, include_gates=True)
            gt_room_state = gt_local_br
        elif map_type_lower == "rooms":
            per_room: List[CogMapMetrics] = []
            for rid in sorted(gt_rooms_map.keys()):
                gt_br = gt_rooms_map[rid]
                if len(gt_br.objects) == 0:
                    continue
                pred_br = pred_rooms_map.get(str(rid)) or pred_rooms_map.get(rid) or BaseRoom(objects=[], name=f"pred_room_{rid}")
                per_room.append(self._compare_baserooms(pred_br, gt_br))
            metrics = CogMapMetrics.average(per_room)
            pred_json = {rid: self.baseroom_to_json(pred_rooms_map.get(str(rid)) or BaseRoom(objects=[], name=f"pred_room_{rid}"), include_gates=False) for rid in gt_rooms_map.keys()}
            gt_room_state = None
        else:
            raise ValueError(f"Invalid map type: {map_type_lower}")

        connectivity_block: Dict[str, float] = {}
        if map_type_lower == "global":
            gate_acc = self._evaluate_gate_connections(pred_gates, gt_room)
            connectivity_block = {"conn_acc": float(gate_acc)} if isinstance(gate_acc, (int, float)) else {}

        # Populate GT jsons and room states
        gt_json_out: Dict[str, Any] = {}
        gt_json_full: Dict[str, Any] = {}
        gt_room_state_full: Optional[BaseRoom] = None
        pred_rooms_state: Dict[str, BaseRoom] = {}
        gt_rooms_state: Dict[str, BaseRoom] = {}

        if map_type_lower == "global":
            # observed-only GT json (already filtered):
            gt_json_out = self.baseroom_to_json(gt_global_br, include_gates=True)
            # full GT json and room state (all objects):
            full_raw = self._baseroom_from_gt(gt_room, gt_agent)
            full_global = _transform_baseroom(full_raw, gt_agent.init_pos, gt_agent.init_ori)
            gt_room_state_full = full_global
            gt_json_full = self.baseroom_to_json(full_global, include_gates=True)
        elif map_type_lower == "local":
            gt_json_out = self.baseroom_to_json(gt_room_state, include_gates=True) if gt_room_state else {}
        elif map_type_lower == "rooms":
            for rid, gt_br in gt_rooms_map.items():
                gt_rooms_state[str(rid)] = gt_br
            for rid, pr in pred_rooms_map.items():
                pred_rooms_state[str(rid)] = pr

        if map_type_lower == "global":
            # compute full metrics w.r.t. full GT
            metrics_full = self._compare_baserooms(pred_global_br, gt_room_state_full or pred_global_br)
            return GlobalCogMapTurnLog(
                type=map_type_lower,
                extraction_success=True,
                original_response=assistant_response,
                pred_json=pred_json,
                pred_room_state=pred_state,
                metrics=metrics,
                connectivity=connectivity_block,
                gt_room_state=gt_room_state,
                gt_json=gt_json_out,
                gt_room_state_full=gt_room_state_full,
                gt_json_full=gt_json_full,
                metrics_full=metrics_full,
            )
        elif map_type_lower == "local":
            return LocalCogMapTurnLog(
                type=map_type_lower,
                extraction_success=True,
                original_response=assistant_response,
                pred_json=pred_json,
                pred_room_state=pred_state,
                metrics=metrics,
                gt_room_state=gt_room_state,
                gt_json=gt_json_out,
            )
        else:
            # rooms
            return RoomsCogMapTurnLog(
                type=map_type_lower,
                extraction_success=True,
                original_response=assistant_response,
                pred_json=pred_json,
                pred_room_state=None,
                metrics=metrics,
                pred_rooms_state=pred_rooms_state,
                gt_rooms_state=gt_rooms_state,
            )

    def evaluate_cogmaps(self, responses_by_type: Dict[str, str], gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], mode: str) -> CognitiveMapTurnLog:
        """Evaluate multiple types and record one aggregate log for the turn."""
        out = CognitiveMapTurnLog()
        for map_type_key, resp in (responses_by_type or {}).items():
            if not isinstance(resp, str):
                continue
            single = self.evaluate_cogmap_type(resp, gt_room, gt_agent, observed_items, map_type_key, mode)
            if single is None:
                continue
            if single.type == "global":
                out.global_log = single
            elif single.type == "local":
                out.local_log = single
            elif single.type == "rooms":
                out.rooms_log = single
        turn_log = out
        if mode == "explore":
            self.explore_logs.append(turn_log)
        else:
            self.evaluate_log = turn_log
        return turn_log
            
    
    def get_cogmap_summary(self) -> Dict[str, Any]:
        """Get metrics for both explore and evaluate modes (per type)."""
        def _extract(log: Optional[CognitiveMapTurnLog]) -> Dict[str, Any]:
            empty = {m: 0.0 for m in ("dir", "facing", "pos", "overall")}
            out = {"global": empty.copy(), "local": empty.copy(), "rooms": empty.copy()}
            if log is None or not isinstance(log, CognitiveMapTurnLog):
                return out
            if getattr(log, 'global_log', None) and log.global_log.metrics.valid:
                out["global"] = log.global_log.metrics.to_dict()
            if getattr(log, 'local_log', None) and log.local_log.metrics.valid:
                out["local"] = log.local_log.metrics.to_dict()
            if getattr(log, 'rooms_log', None) and log.rooms_log.metrics.valid:
                out["rooms"] = log.rooms_log.metrics.to_dict()
            return out
        explore = _extract(self.explore_logs[-1] if self.explore_logs else None)
        evaluate = _extract(self.evaluate_log)
        return {"explore": explore, "evaluate": evaluate}
    
    @staticmethod
    def _calculate_cogmap_per_turn(env_data_list: List[Dict], mode: str = "update") -> Dict[str, List[float]]:
        """Calculate average cognitive map metrics for each turn across all samples.

        Args:
            env_data_list: List of environment data dictionaries
            mode: Either "update" or "full" to specify which cognitive map mode to use

        Returns:
            Dict with only 'global' key containing dict of metric lists
        """
        from collections import defaultdict

        # Collect all turn metrics by turn index for global level only
        turn_metrics = {
            'global': defaultdict(lambda: defaultdict(list))  # turn_idx -> metric -> values
        }
        PAD = 0.0

        for env_data in env_data_list:
            env_turn_logs = env_data.get('env_turn_logs', [])
            for turn_idx, turn_log in enumerate(env_turn_logs):
                if not turn_log['is_exploration_phase']:
                    continue
                # Select the per-type log aggregator
                cogmap_agg = turn_log.get('cogmap_log', {})
                for level in ['global', 'local', 'rooms']:
                    level_data = (cogmap_agg.get(level, {}) or cogmap_agg.get('by_type', {}).get(level, {}).get('metrics', {}))
                    for metric in ['dir', 'facing', 'pos', 'overall']:
                        value = level_data.get(metric) if isinstance(level_data, dict) else None
                        if value is not None and isinstance(value, (int, float)):
                            turn_metrics[level][turn_idx][metric].append(float(value))

        # Calculate averages for each turn and level
        result = {'global': {}, 'local': {}, 'rooms': {}}

        for level in ['global', 'local', 'rooms']:
            level_turn_metrics = turn_metrics[level]
            max_turns = max(level_turn_metrics.keys()) if level_turn_metrics else -1

            for metric in ['dir', 'facing', 'pos', 'overall']:
                avg_values = []

                for turn_idx in range(max_turns + 1):
                    if (turn_idx in level_turn_metrics and metric in level_turn_metrics[turn_idx] and level_turn_metrics[turn_idx][metric]):
                        values = level_turn_metrics[turn_idx][metric]
                        avg_value = sum(values) / len(values)
                        avg_values.append(avg_value)
                    else:
                        avg_values.append(PAD)

                result[level][metric] = avg_values
        return result

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict] = None) -> Dict[str, Any]:
        """Calculate cognitive map performance from env_data_list."""
        assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"

        metrics = ["dir", "facing", "pos", "overall"]

        # Calculate metrics from cogmap logs of each sample
        explore_metrics = {m: [] for m in metrics}
        evaluate_metrics_by_task = {}  # task_type -> {metric -> []}

        for env_data in env_data_list:
            env_turn_logs = env_data.get('env_turn_logs', [])

            # Find the last exploration turn with cogmap_full_log
            last_explore_cogmap = None
            for turn_log in reversed(env_turn_logs):
                if turn_log.get('is_exploration_phase', False):
                    cogmap_full_log = turn_log.get('cogmap_full_log', {})
                    if cogmap_full_log:
                        last_explore_cogmap = cogmap_full_log.get('global', {})
                        for metric in metrics:
                            value = last_explore_cogmap.get(metric)
                            if value is not None:
                                explore_metrics[metric].append(value)
                        break

            # Get evaluation task cogmap metrics - store separately for each task
            evaluation_tasks = env_data.get('evaluation_tasks', {})
            for task_type, task_data in evaluation_tasks.items():
                if task_type not in evaluate_metrics_by_task:
                    evaluate_metrics_by_task[task_type] = {m: [] for m in metrics}

                cogmap_full_log = task_data.get('cogmap_full_log', {})
                if cogmap_full_log:
                    global_metrics = cogmap_full_log.get('global', {})
                    for metric in metrics:
                        value = global_metrics.get(metric)
                        if value is not None:
                            evaluate_metrics_by_task[task_type][metric].append(value)

        # Calculate averages, excluding None values
        explore_avg = {}
        evaluate_avg_by_task = {}

        for metric in metrics:
            if explore_metrics[metric]:  # Only add if there are values
                explore_avg[metric] = sum(explore_metrics[metric]) / len(explore_metrics[metric])

        # Calculate averages for each evaluation task separately
        for task_type, task_metrics in evaluate_metrics_by_task.items():
            task_avg = {}
            for metric in metrics:
                if task_metrics[metric]:  # Only add if there are values
                    task_avg[metric] = sum(task_metrics[metric]) / len(task_metrics[metric])

            # Only add task if it has any metrics
            if task_avg:
                evaluate_avg_by_task[task_type] = task_avg

        out = {}

        # Only add explore section if it has metrics
        if explore_avg:
            out["explore"] = {"global": explore_avg}

        # Only add evaluate section if it has task metrics
        if evaluate_avg_by_task:
            out["evaluate"] = {"global": evaluate_avg_by_task}

        # Calculate average cogmap per turn across all samples
        cogmap_update_per_turn = CognitiveMapManager._calculate_cogmap_per_turn(env_data_list, mode="update")
        cogmap_full_per_turn = CognitiveMapManager._calculate_cogmap_per_turn(env_data_list, mode="full")
        out["cogmap_update_per_turn"] = cogmap_update_per_turn
        out["cogmap_full_per_turn"] = cogmap_full_per_turn

        return out
    
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
        try:
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
        except Exception:
            return None

    def _parse_predicted_maps(self, json_data: Dict[str, Any]) -> Tuple[Optional[BaseRoom], Optional[BaseRoom], Dict[str, BaseRoom], Dict[str, Any]]:
        """Return (global_br, local_br, rooms_map, gates_dict)."""
        # If no explicit 'global', assume flat map is the global section
        if any(k in json_data for k in ("global", "local", "rooms", "gates")):
            global_sec = json_data.get('global')
        else:
            global_sec = json_data
        pred_global_br = self._parse_section_to_baseroom(global_sec, "pred_global") if isinstance(global_sec, dict) else None

        local_sec = json_data.get('local') if isinstance(json_data, dict) else None
        pred_local_br = self._parse_section_to_baseroom(local_sec, "pred_local") if isinstance(local_sec, dict) else None

        rooms_map: Dict[str, BaseRoom] = {}
        rooms_sec = json_data.get('rooms') if isinstance(json_data, dict) else None
        if isinstance(rooms_sec, dict):
            for rid, sec in rooms_sec.items():
                if isinstance(sec, dict):
                    br = self._parse_section_to_baseroom(sec, f"pred_room_{rid}")
                    if br is not None:
                        rooms_map[str(rid)] = br

        gates_sec = json_data.get('gates') if isinstance(json_data, dict) else None
        gates_dict = gates_sec if isinstance(gates_sec, dict) else {}
        return pred_global_br, pred_local_br, rooms_map, gates_dict

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
        objs.append(Object(name='agent', pos=gt_agent.pos.copy(), ori=gt_agent.ori.copy(), has_orientation=True))
        return BaseRoom(objects=objs, name='gt')

    def _build_gt_global_baseroom(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = _transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = set(observed_set) | {"agent"}
        return self._filter_br_by_names(br, keep)

    def _build_gt_local_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return _transform_baseroom(raw, gt_agent.pos, gt_agent.ori)

    def _build_gt_local_with_gates(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        for g in gt_room.gates:
            objs.append(Object(name=g.name, pos=g.pos.copy(), ori=g.ori.copy(), has_orientation=True))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return _transform_baseroom(raw, gt_agent.pos, gt_agent.ori)
    
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
            out[int(rid)] = _transform_baseroom(BaseRoom(objects=objs, name=f'gt_room_{rid}'), anchor_pos, anchor_ori)
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

    def _compare_baserooms(self, pred_room: BaseRoom, gt_room: BaseRoom) -> CogMapMetrics:
        dir_sim = self._calculate_dir_sim(pred_room, gt_room)
        facing_sim = self._calculate_facing_sim(pred_room, gt_room)
        pos_sim = self._calculate_pos_sim(pred_room, gt_room, allow_scale=bool(self.config.get('pos_allow_scale', True)))
        overall_sim = 0.5 * dir_sim + 0.2 * facing_sim + 0.3 * pos_sim
        return CogMapMetrics(dir=dir_sim, facing=facing_sim, pos=pos_sim, overall=overall_sim)

    def _calculate_dir_sim(self, pred_room: BaseRoom, gt_room: BaseRoom) -> float:
        """Pairwise allocentric bin agreement over GT object pairs.

        Missing predicted objects are counted as incorrect pairs.
        """
        pred = {o.name: o for o in pred_room.objects}
        gt = {o.name: o for o in gt_room.objects}
        names = sorted(gt.keys())
        if len(names) < 2:
            return 1.0
        bin_system = CardinalBinsAllo()
        tot = cor = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = gt[names[i]], gt[names[j]]
                gt_rel = PairwiseRelationshipDiscrete.relationship(a.pos, b.pos, None, bin_system)
                p1, p2 = pred.get(names[i]), pred.get(names[j])
                if p1 is not None and p2 is not None:
                    pr = PairwiseRelationshipDiscrete.relationship(p1.pos, p2.pos, None, bin_system)
                    if pr.direction.bin_id == gt_rel.direction.bin_id:
                        cor += 1.0
                tot += 1.0
        return cor / tot if tot else 0.0

    def _calculate_facing_sim(self, pred_room: BaseRoom, gt_room: BaseRoom) -> float:
        pred = {o.name: o for o in pred_room.objects}
        gt = {o.name: o for o in gt_room.objects}
        names = sorted(gt.keys())
        tot = cor = 0.0
        for name in names:
            g = gt[name]
            if not g.has_orientation:
                continue
            p = pred.get(name)
            tot += 1.0
            if p is not None and np.array_equal(p.ori, g.ori):
                cor += 1.0
        return cor / tot if tot else 1.0

    def _calculate_pos_sim(self, pred_room: BaseRoom, gt_room: BaseRoom, allow_scale: bool = True) -> float:
        """Position similarity with optional scale alignment and coverage penalty.

        Given matched points P_pred and P_gt (same name ordering):
        - If allow_scale: find s* that minimizes ||s*·P_pred − P_gt|| in least squares
          s* = (Σ r_i·e_i) / (Σ e_i·e_i), where e_i from pred, r_i from gt
        - RMSE = sqrt(mean(||s*·e_i − r_i||^2))
        - Normalize by a global L computed once and convert to similarity via exp(−RMSE/L)

        Similarity is exp(-RMSE/L) scaled by coverage = (#matched GT objects)/(#GT objects).
        """
        pred = {o.name: o for o in pred_room.objects}
        gt = {o.name: o for o in gt_room.objects}
        gt_names = sorted(gt.keys())
        matched = [n for n in gt_names if n in pred]
        if len(matched) == 0 or len(gt_names) == 0:
            return 1.0
        P1 = np.array([pred[n].pos for n in matched], dtype=float)
        P2 = np.array([gt[n].pos for n in matched], dtype=float)
        if allow_scale:
            den = float((P1 * P1).sum())
            if den == 0.0:
                return 0.0
            scale = float((P2 * P1).sum()) / den
        else:
            scale = 1.0
        rmse = np.sqrt(((P1 * scale - P2) ** 2).sum(axis=1).mean())
        L = float(self._pos_norm_L or 0.0)
        base = float(np.exp(-rmse / L)) if L > 0 else 0.0
        coverage = float(len(matched)) / float(len(gt_names))
        return base * coverage

    # =============================== Consistency helpers ===============================
    @staticmethod
    def _names_set(br: Optional[BaseRoom]) -> set[str]:
        if br is None:
            return set()
        return {o.name for o in br.objects} if br else set()

    def _restrict_br_to_names(self, br: Optional[BaseRoom], names: set[str], name: str) -> BaseRoom:
        """Return a shallow BaseRoom copy with only objects whose names are in `names`."""
        if br is None or not names:
            return BaseRoom(objects=[], name=f"{name}_empty")
        keep = [o for o in br.objects if o.name in names]
        return BaseRoom(objects=keep, name=name)

    def _compare_on_common_subset(self, a: Optional[BaseRoom], b: Optional[BaseRoom]) -> CogMapMetrics:
        """Compute dir/facing/pos similarity using only objects present in BOTH rooms."""
        if a is None or b is None:
            return CogMapMetrics.invalid()
        names = self._names_set(a) & self._names_set(b)
        if not names:
            return CogMapMetrics.invalid()
        a_sub = self._restrict_br_to_names(a, names, a.name)
        b_sub = self._restrict_br_to_names(b, names, b.name)
        return self._compare_baserooms(a_sub, b_sub)

    def _consistency_local_vs_global(
        self,
        pred_local_br: Optional[BaseRoom],
        pred_global_br: Optional[BaseRoom],
        gt_agent: Agent
    ) -> CogMapMetrics:
        """
        Transform predicted local into the initial frame, then compare to predicted global
        on their common object subset.
        """
        anchor_pos = np.array(gt_agent.pos, dtype=float)
        anchor_ori = np.array(gt_agent.ori, dtype=int)
        # local(anchor) -> world -> initial
        local_in_initial = _br_from_anchor_to_initial(pred_local_br, anchor_pos, anchor_ori, gt_agent)
        # compare on common subset
        return self._compare_on_common_subset(local_in_initial, pred_global_br)

    def _consistency_rooms_vs_global(
        self,
        pred_rooms_map: Dict[str, BaseRoom],
        pred_global_br: Optional[BaseRoom],
        gt_agent: Agent,
        gt_room: Room
    ) -> tuple[CogMapMetrics, Dict[str, Dict[str, float]]]:
        """
        For each predicted room section:
        1) transform room map into the initial frame,
        2) compare against predicted global on the common subset,
        then return the average metrics and the per-room metrics dict.
        """
        per_room_metrics: List[CogMapMetrics] = []
        per_room_out: Dict[str, Dict[str, float]] = {}
        if pred_global_br is None:
            return CogMapMetrics.invalid(), per_room_out

        # Sort keys to keep output stable; tolerate str/int room IDs
        for rid, room_br in sorted(
            pred_rooms_map.items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
        ):
            gate_name = None
            gate_name = self.entry_gate_by_room.get(int(rid))
            if gate_name is None:
                # no entry gate recorded → skip consistency for this room
                per_room_out[rid] = {}
                continue
            g = next((gg for gg in gt_room.gates if gg.name == gate_name), None)
            if g is None:
                per_room_out[rid] = {}
                continue
            gate_pos = np.array(g.pos, dtype=float)
            gate_ori = g.get_ori_for_room(int(rid))
            room_in_initial = _br_from_anchor_to_initial(room_br, gate_pos, gate_ori, gt_agent)
            m = self._compare_on_common_subset(room_in_initial, pred_global_br)
            if m.valid:
                per_room_metrics.append(m)
                per_room_out[rid] = m.to_dict()
            else:
                per_room_out[rid] = {}
        avg_m = CogMapMetrics.average(per_room_metrics) if per_room_metrics else CogMapMetrics.invalid()
        return avg_m, per_room_out

    # =============================== Room entry tracking =============================== 
    def register_room_entry(self, room_id: int, gate_name: str) -> None:
        """Record the first gate used to enter a room."""
        rid = int(room_id)
        if rid not in self.entry_gate_by_room:
            self.entry_gate_by_room[rid] = gate_name

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
        br = _transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
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