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
    DegreeRel,
)


COGMAP_INSTRUCTION_SHORTER = """\
## Cognitive Map (multi-map)

Keep a concise multi-map JSON of the scene on a N by M grid.

- Global: origin [0,0] and +Y is your initial facing direction
- Local: origin at your current pose; +Y is forward
- Rooms: each room uses its entry gate as origin; +Y points into that room
- Gates: list connections as room id pairs (connects: [room_id_a, room_id_b])

Fields:
- position: [x, y] in the map’s coordinate system (integers or integer-like)
- facing: one of "north|south|east|west" (omit or set unknown if not applicable)
- confidence: "high" (certain), "medium" (estimated), "low" (unknown)

Content rules:
- global: include observed objects and gates; include agent; exclude "initial_pos"
- local: include visible objects only; exclude agent
- rooms: include observed objects in that room only; exclude origin gate and agent
- gates: each entry contains **only** `"connects": [room_id_a, room_id_b]`.

**Important**:
- `facing` and `confidence` **must always be present** for all objects and agents, even in `local` and `rooms`.
- You must always use one of "north|south|east|west" for facings in global, local, and rooms
Always output the cognitive map JSON first in your thinking. Include at least `global`; `local`, `rooms`, and `gates` are optional, but if included, they must follow the schema strictly.


Example:
```json
{{
  "global": {{
    "agent": {{"position": [0, 0], "facing": "north", "confidence": "high"}},
    "table": {{"position": [2, 1], "facing": "east", "confidence": "medium"}}
  }},
  "local": {{
    "agent": {{"position": [0, 0], "facing": "north", "confidence": "high"}},
    "chair": {{"position": [-1, 2], "facing": "west", "confidence": "high"}}
  }},
  "rooms": {{
    "1": {{"sofa": {{"position": [1, 0], "facing": "south", "confidence": "high"}}}}
  }},
  "gates": {{
    "door_0": {{"connects": [1, 2]}}
  }}
}}
```
"""

COGMAP_INSTRUCTION_GLOBAL_ONLY = """\
## Cognitive Map (global only)

Keep a concise global map JSON of the scene on a N by M grid.

- Global: origin [0,0] and +Y is your initial facing direction

Fields:
- position: [x, y] in the map’s coordinate system (integers or integer-like)
- facing: one of "north|south|east|west" (omit or set unknown if not applicable)
- confidence: "high" (certain), "medium" (estimated), "low" (unknown)

Content rules:
- Global: include observed objects and gates; include agent; exclude "initial_pos"

Only include the `global` section (you may also include `gates`).
Always output the cognitive map JSON first in your thinking.

Example:
```json
{{
  "global": {{
    "agent": {{"position": [0, 0], "facing": "north", "confidence": "high"}},
    "table": {{"position": [2, 1], "facing": "east", "confidence": "medium"}}
  }}
}}
"""

COGMAP_EXP_REQUIRED_INSTRUCTION = """
In your thinking (<think> ... </think>):
1) Briefly reason about your cognitive map
2) Output the cognitive map JSON
3) Then reason about exploration and provide only the <answer>...</answer>

Example:
```json
{{
  "global": {{
    "agent": {{"position": [0, 0], "facing": "north", "confidence": "high"}},
  }},
}}
```
"""

COGMAP_EVAL_REQUIRED_INSTRUCTION = """
In your thinking (<think> ... </think>):
1) Briefly reason about your cognitive map
2) Output the cognitive map JSON
3) Then reason about the question and provide only the <answer>...</answer>

Example:
```json
{{
  "global": {{
    "agent": {{"position": [0, 0], "facing": "north", "confidence": "high"}},
  }},
}}
```
"""


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
class CognitiveMapTurnLog:
    """Per-turn metrics for cognitive map evaluation."""
    # Hierarchical metrics
    global_metrics: CogMapMetrics = field(default_factory=CogMapMetrics)
    local_metrics: CogMapMetrics = field(default_factory=CogMapMetrics)
    rooms_metrics: CogMapMetrics = field(default_factory=CogMapMetrics)
    gates: Dict[str, float] = field(default_factory=dict)
    # Extraction status
    extraction_success: bool = False
    # Optional for debugging/inspection (predicted global map as BaseRoom)
    pred_room_state: Optional['BaseRoom'] = None
    # Ground Truth 
    gt_global_cog: Dict[str, Any] = field(default_factory=dict)
    gt_local_cog: Dict[str, Any] = field(default_factory=dict)
    gt_rooms_cog: Dict[str, Any] = field(default_factory=dict)
    gt_gates: Dict[str, Any] = field(default_factory=dict)
    # Backward-compatible flat fields reflecting global metrics
    dir_sim: float = 0.0
    facing_sim: float = 0.0
    pos_sim: float = 0.0
    overall_sim: float = 0.0

    def to_dict(self):
        return {
            "global": (self.global_metrics.to_dict() if self.global_metrics.valid else {}),
            "local": (self.local_metrics.to_dict() if self.local_metrics.valid else {}),
            "rooms": (self.rooms_metrics.to_dict() if self.rooms_metrics.valid else {}),
            "gates": (self.gates or {}),
            "extraction_success": self.extraction_success,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {},
            # GT
            "gt_global_cog": (self.gt_global_cog or {}),
            "gt_local_cog": (self.gt_local_cog or {}),
            "gt_rooms_cog": (self.gt_rooms_cog or {}),
            "gt_gates": (self.gt_gates or {}),
            # flat for backward-compatibility
            "dir_sim": self.dir_sim,
            "facing_sim": self.facing_sim,
            "pos_sim": self.pos_sim,
            "overall_sim": self.overall_sim,
        }


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
        if getattr(obj, 'has_orientation', True):
            o = _transform_ori(obj.ori, anchor_ori)
        objects.append(Object(name=obj.name, pos=p, ori=o, has_orientation=getattr(obj, 'has_orientation', True)))
    return BaseRoom(objects=objects, name=getattr(room, 'name', 'room'))


class CognitiveMapManager:
    """Evaluate cognitive map JSON against ground truth."""
    
    DEFAULT_COGMAP_SUMMARY = {
        "global": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
        "local": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
        "rooms": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
        "gates": {"conn_acc": 0.0},
        "extraction_success_rate": 0.0,
        "n_successful": 0,
        "n_evaluations": 0,
    }
    
    def __init__(self, cogmap_type: str = "standard", pos_allow_scale: bool = True, scope: str = "all"):
        """Initialize cognitive map manager."""
        self.turn_logs: List[CognitiveMapTurnLog] = []
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

    def get_cognitive_map_instruction(self) -> str:
        assert self.config['cogmap_type'] == "standard", "Only standard format is supported"
        if self.config.get("scope") == "global":
            return COGMAP_INSTRUCTION_GLOBAL_ONLY
        return COGMAP_INSTRUCTION_SHORTER
        
    def evaluate_cognitive_map(self, assistant_response: str, gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Extract JSON and evaluate global/local/rooms/gates.

        All comparisons are between BaseRooms in the same coordinate system.
        - Global: GT transformed using agent initial pose as origin
        - Local: GT filtered by FOV and transformed using current pose
        - Rooms: GT per-room transformed using entry gate as origin
        - Gates: compare connectivity lists
        """
        json_data = self._extract_json_from_text(assistant_response)
        if json_data is None or gt_room is None:
            self.turn_logs.append(CognitiveMapTurnLog(extraction_success=False))
            return None

        # Observed/visible sets
        all_item_names = {o.name for o in getattr(gt_room, 'all_objects', [])}
        observed_set: set[str] = set(all_item_names if observed_items is None else [str(x) for x in observed_items])
        visible_names = self._visible_object_names(gt_room, gt_agent)

        # Preprocess predicted JSON then parse
        json_data = self._preprocess_predicted(json_data, observed_set, visible_names, gt_room)
        pred_global_br, pred_local_br, pred_rooms_map, pred_gates = self._parse_predicted_maps(json_data)

        # Ensure predicted sections exist (empty) where needed
        pred_global_br = pred_global_br or BaseRoom(objects=[], name="pred_global")
        pred_local_br = pred_local_br or BaseRoom(objects=[], name="pred_local")
        pred_rooms_map = pred_rooms_map or {}
        pred_gates = pred_gates or {}

        # Build GT BaseRooms (already filtered)
        gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
        gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
        gt_rooms_map = self._build_gt_room_baserooms(gt_room, gt_agent, observed_set)

        # Ensure global position normalization is set
        self._ensure_pos_norm_L(gt_room, gt_agent)

        # Evaluate global (used for overall summary)
        global_m = self._compare_baserooms(pred_global_br, gt_global_br)

        # Evaluate local
        local_m = CogMapMetrics.invalid()
        if self.config.get("scope") == "all" and len(getattr(gt_local_br, 'objects', [])) > 0:
            local_m = self._compare_baserooms(pred_local_br, gt_local_br)

        # Evaluate rooms (average across GT rooms; missing pred treated empty)
        rooms_m = CogMapMetrics.invalid()
        if self.config.get("scope") == "all":
            per_room: List[CogMapMetrics] = []
            print("Hiiii")
            for rid in sorted(gt_rooms_map.keys()):
                gt_br = gt_rooms_map[rid]
                if len(getattr(gt_br, 'objects', [])) == 0:
                    continue
                pred_br = pred_rooms_map.get(str(rid)) or pred_rooms_map.get(rid) or BaseRoom(objects=[], name=f"pred_room_{rid}")
                per_room.append(self._compare_baserooms(pred_br, gt_br))
            rooms_m = CogMapMetrics.average(per_room)

        # Evaluate gates connectivity
        gate_acc: Optional[float] = None
        if self.config.get("scope") == "all":
            gate_acc = self._evaluate_gate_connections(pred_gates, gt_room)

        metrics = {
            "global": (global_m.to_dict() if global_m.valid else {}),
            "local": (local_m.to_dict() if local_m.valid else {}),
            "rooms": (rooms_m.to_dict() if rooms_m.valid else {}),
            "gates": ({"conn_acc": float(gate_acc)} if isinstance(gate_acc, (int, float)) else {}),
        }
        # Gather GT gate connectivity for logging
        gt_gates_dict = self._gt_gate_connections_dict(gt_room)
        # Log all results (global fields kept for summary compatibility)
        turn_log = CognitiveMapTurnLog(
            global_metrics=global_m,
            local_metrics=local_m,
            rooms_metrics=rooms_m,
            gates=({"conn_acc": float(gate_acc)} if isinstance(gate_acc, (int, float)) else {}),
            extraction_success=True,
            pred_room_state=pred_global_br,
            # GT json
            gt_global_cog = self.baseroom_to_json(gt_global_br, include_gates=True) if gt_global_br else {},
            gt_local_cog = self.baseroom_to_json(gt_local_br, include_gates=False) if gt_local_br else {},      
            gt_rooms_cog = {rid: self.baseroom_to_json(br, include_gates=False) for rid, br in (gt_rooms_map or {}).items()},
            gt_gates=gt_gates_dict,
            dir_sim=global_m.dir,
            facing_sim=global_m.facing,
            pos_sim=global_m.pos,
            overall_sim=global_m.overall,
        )
        self.turn_logs.append(turn_log)
        return metrics
            
    
    def get_cogmap_summary(self) -> Dict[str, Any]:
        """Get cognitive map summary statistics (hierarchical)."""
        if not self.turn_logs:
            out = copy.deepcopy(self.DEFAULT_COGMAP_SUMMARY)
            out["n_evaluations"] = 0
            return out
        
        successful_logs = [log for log in self.turn_logs if log.extraction_success]
        n_evaluations = len(self.turn_logs)
        n_successful = len(successful_logs)
        success_rate = n_successful / n_evaluations if n_evaluations else 0.0
        
        if n_successful == 0:
            out = copy.deepcopy(self.DEFAULT_COGMAP_SUMMARY)
            out.update({"n_evaluations": n_evaluations, "extraction_success_rate": success_rate})
            return out

        g_vals = [l.global_metrics for l in successful_logs if l.global_metrics.valid]
        l_vals = [l.local_metrics for l in successful_logs if l.local_metrics.valid]
        r_vals = [l.rooms_metrics for l in successful_logs if l.rooms_metrics.valid]
        g_avg = CogMapMetrics.average(g_vals).to_dict() if g_vals else self.DEFAULT_COGMAP_SUMMARY["global"]
        l_avg = CogMapMetrics.average(l_vals).to_dict() if l_vals else self.DEFAULT_COGMAP_SUMMARY["local"]
        r_avg = CogMapMetrics.average(r_vals).to_dict() if r_vals else self.DEFAULT_COGMAP_SUMMARY["rooms"]
        gate_list = [float((l.gates or {}).get("conn_acc", 0.0)) for l in successful_logs]
        gate_avg = float(np.mean(gate_list)) if gate_list else 0.0
        
        return {
            "global": g_avg,
            "local": l_avg,
            "rooms": r_avg,
            "gates": {"conn_acc": gate_avg},
            "extraction_success_rate": success_rate,
            "n_successful": n_successful,
            "n_evaluations": n_evaluations,
        }
    
    @staticmethod
    def aggregate_group_performance(cogmap_summaries: List[Dict]) -> Dict[str, float]:
        """Average hierarchical metrics across summaries."""
        if not cogmap_summaries:
            return {
                "global": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
                "local": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
                "rooms": {"dir": 0.0, "facing": 0.0, "pos": 0.0, "overall": 0.0},
                "gates": {"conn_acc": 0.0},
                "overall_avg_extraction_success_rate": 0.0,
                "overall_avg_evaluations": 0.0,
                "overall_n_successful": 0,
            }

        def avg_key(path: List[str], default: float = 0.0) -> float:
            vals = []
            for s in cogmap_summaries:
                cur = s
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, (int, float)):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else default

        out = {
            "global": {m: avg_key(["global", m]) for m in ("dir", "facing", "pos", "overall")},
            "local": {m: avg_key(["local", m]) for m in ("dir", "facing", "pos", "overall")},
            "rooms": {m: avg_key(["rooms", m]) for m in ("dir", "facing", "pos", "overall")},
            "gates": {"conn_acc": avg_key(["gates", "conn_acc"])},
            "overall_avg_extraction_success_rate": avg_key(["extraction_success_rate"]),
            "overall_avg_evaluations": avg_key(["n_evaluations"]),
            "overall_n_successful": sum(int(s.get("n_successful", 0)) for s in cogmap_summaries),
        }
        return out
    

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
        for o in getattr(gt_room, 'objects', []):
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        # include gates
        for g in getattr(gt_room, 'gates', []):
            objs.append(Object(name=g.name, pos=g.pos.copy(), ori=g.ori.copy(), has_orientation=True))
        # include agent
        objs.append(Object(name='agent', pos=gt_agent.pos.copy(), ori=gt_agent.ori.copy(), has_orientation=True))
        return BaseRoom(objects=objs, name='gt')

    def _build_gt_global_baseroom(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = _transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        gate_names = {g.name for g in getattr(gt_room, 'gates', [])}
        keep = set(observed_set) | gate_names | {"agent"}
        return self._filter_br_by_names(br, keep)

    def _build_gt_local_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return _transform_baseroom(raw, gt_agent.pos, gt_agent.ori)

    def _build_gt_room_baserooms(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> Dict[int, BaseRoom]:
        out: Dict[int, BaseRoom] = {}
        if not isinstance(gt_room, Room):
            return out
        # Helper: get the entry gate for a room id (prefer manager-registered entry gate, else first gate in that room)
        def _get_entry_gate_for_room(rid: int) -> Gate | None:
            # 1) user-registered first-entry gate
            name = self.entry_gate_by_room.get(int(rid))
            if name:
                g = next((g for g in getattr(gt_room, "gates", []) if g.name == name), None)
                if g is not None:
                    return g
            # 2) fallback to first gate that connects to this room (from room’s structure)
            gb = getattr(gt_room, "gates_by_room", {})
            cand_names = gb.get(int(rid), []) or gb.get(str(rid), [])  # tolerate string/int keys
            for nm in cand_names:
                g = next((g for g in getattr(gt_room, "gates", []) if g.name == nm), None)
                if g is not None:
                    return g
            return None

        # Helper: get gate orientation that points INTO a given room id
        def _gate_ori_into_room(g: Gate, rid: int) -> np.ndarray:
            # Prefer the Gate method if available
            if hasattr(g, "get_ori_for_room"):
                try:
                    return g.get_ori_for_room(int(rid))
                except Exception:
                    pass
            # Else fallback to ori_by_room mapping (note: keys may be strings in to_dict output)
            ob = getattr(g, "ori_by_room", None)
            if isinstance(ob, dict):
                v = ob.get(int(rid)) or ob.get(str(rid))
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return np.array([int(v[0]), int(v[1])], dtype=int)
            # As a last resort, try the gate's own ori
            return np.array(getattr(g, "ori", [0, 1]), dtype=int)

        # Iterate rooms; tolerate int or str keys
        rooms_map = getattr(gt_room, "objects_by_room", {})
        for rid_key in sorted(rooms_map.keys(), key=lambda x: int(x)):
            rid = int(rid_key)
            entry_gate = _get_entry_gate_for_room(rid)
            if entry_gate is None:
                # No entry gate recorded and none available for this room → skip
                continue

            anchor_pos = np.array(entry_gate.pos, dtype=float)
            anchor_ori = _gate_ori_into_room(entry_gate, rid)

        # Collect observed objects that truly belong to this room id
            objs: List[Object] = []
            for o in getattr(gt_room, "objects", []):
                try:
                    if int(getattr(o, "room_id", -999)) != rid:
                        continue
                except Exception:
                    continue
                if o.name not in observed_set:
                    continue
                # Add object (not gate, not agent — objects list already excludes gates/agent)
                objs.append(
                    Object(
                    name=o.name,
                    pos=np.array(o.pos, dtype=float).copy(),
                    ori=np.array(o.ori, dtype=int).copy(),
                    has_orientation=getattr(o, "has_orientation", True),
                    )
                )

            # Transform to the room's frame (origin at entry gate, +Y into room)
            br = _transform_baseroom(BaseRoom(objects=objs, name=f"gt_room_{rid}"), anchor_pos, anchor_ori)
            out[rid] = br
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
            return 0.0
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
            if not getattr(g, 'has_orientation', False):
                continue
            p = pred.get(name)
            tot += 1.0
            if p is not None and np.array_equal(p.ori, g.ori):
                cor += 1.0
        return cor / tot if tot else 0.0

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
            return 0.0
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
        for o in getattr(gt_room, 'objects', []):
            if BaseAction._is_visible(gt_agent, o):
                names.add(o.name)
        return names

    def _preprocess_predicted(self, json_data: Dict[str, Any], observed: set[str], visible: set[str], gt_room: Room) -> Dict[str, Any]:
        jd = copy.deepcopy(json_data) if isinstance(json_data, dict) else {}
        gate_names = {g.name for g in getattr(gt_room, 'gates', [])}

        # Global: keep observed objects and gates; include agent; drop initial_pos
        if isinstance(jd.get('global'), dict):
            g = jd['global']
            keep = set(observed) | gate_names | {"agent"}
            jd['global'] = {k: v for k, v in g.items() if k in keep and k != 'initial_pos'}

        # Local: keep visible objects only; exclude agent and gates implicitly
        if isinstance(jd.get('local'), dict):
            loc = jd['local']
            jd['local'] = {k: v for k, v in loc.items() if k in visible}

        # Rooms: per-room observed, non-gate objects only
        rooms = jd.get('rooms') if isinstance(jd, dict) else None
        if isinstance(rooms, dict):
            out_rooms = {}
            for rid, sec in rooms.items():
                if not isinstance(sec, dict):
                    continue
                keep = {n for n in observed if n in gt_room.room_by_object and gt_room.room_by_object[n] == int(rid)}
                out_rooms[str(rid)] = {k: v for k, v in sec.items() if k in keep}
            jd['rooms'] = out_rooms
        return jd

    def _ensure_pos_norm_L(self, gt_room: Room, gt_agent: Agent) -> None:
        if self._pos_norm_L is not None:
            return
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = _transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = {o.name for o in getattr(gt_room, 'objects', [])}
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
        for g in getattr(gt_room, 'gates', []) or []:
            try:
            # expect Gate.room_id like [a,b]
                if isinstance(g.room_id, (list, tuple)) and len(g.room_id) == 2:
                    out[g.name] = {"connects": [int(g.room_id[0]), int(g.room_id[1])]}
            except Exception:
                continue
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
    # Build two rooms connected by door_0
    objs = [
        Object("chair", np.array([1, 1]), np.array([0, 1])),
        Object("table", np.array([3, 3]), np.array([1, 0]))
    ]
    gates = [
        Gate(
        name="door_0", pos=np.array([2, 0]), ori=np.array([0, 1]),
        room_id=[1, 2], ori_by_room={1: np.array([0, 1]), 2: np.array([0, -1])}
        )
    ]
    mask = np.ones((5, 5), dtype=int)
    room = Room(objects=objs, mask=mask, gates=gates)

    agent = Agent(
        name="agent", pos=np.array([0, 0]), ori=np.array([0, 1]),
        room_id=1, init_pos=np.array([0, 0]), init_ori=np.array([0, 1]), init_room_id=1
    )

    # Register entry gate for room 1 and 2
    mgr = CognitiveMapManager()
    mgr.register_room_entry(1, "door_0")
    mgr.register_room_entry(2, "door_0")

    # Observed set includes everything
    observed = {"chair", "table"}

    # Call your function
    gt_rooms = mgr._build_gt_room_baserooms(room, agent, observed)

    # Inspect
    for rid, br in gt_rooms.items():
        print(rid, br)
