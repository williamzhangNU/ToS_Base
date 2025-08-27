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

Keep a concise multi-map JSON of the scene on a {grid_size}×{grid_size} grid.

- Global: origin [0,0] and +Y is your initial facing direction
- Local: origin at your current pose; +Y is forward
- Rooms: each room uses its entry gate as origin; +Y points into that room
- Gates: list connections as room id pairs (gate_name: { connects: [room_id_a, room_id_b] })

Fields:
- position: [x, y] in the map’s coordinate system (integers or integer-like)
- facing: one of "north|south|east|west" (omit or set unknown if not applicable)
- confidence: "high" (certain), "medium" (estimated), "low" (unknown)

Always output the cognitive map JSON first in your thinking. Include at least `global`; `local`, `rooms`, and `gates` are optional.

Example:
```json
{
  "global": {
    "agent": {"position": [0, 0], "facing": "north", "confidence": "high"},
    "table": {"position": [2, 1], "facing": "east", "confidence": "medium"}
  },
  "local": {
    "agent": {"position": [0, 0], "facing": "north", "confidence": "high"},
    "chair": {"position": [-1, 2], "facing": "west", "confidence": "high"}
  },
  "rooms": {
    "1": {"sofa": {"position": [1, 0], "facing": "south", "confidence": "high"}}
  },
  "gates": {
    "door_0": {"connects": [1, 2]}
  }
}
```
"""

COGMAP_EXP_REQUIRED_INSTRUCTION = """
In your thinking (<think> ... </think>):
1) Briefly reason about your cognitive map
2) Output the cognitive map JSON (at least `global`; `local`, `rooms`, `gates` optional)
3) Then reason about exploration and provide only the <answer>...</answer>

Example:
```json
{
  "global": {
    "agent": {"position": [0, 0], "facing": "north", "confidence": "high"},
  },
}
```
"""

COGMAP_EVAL_REQUIRED_INSTRUCTION = """
In your thinking (<think> ... </think>):
1) Briefly reason about your cognitive map
2) Output the cognitive map JSON (at least `global`; `local`, `rooms`, `gates` optional)
3) Then reason about the question and provide only the <answer>...</answer>

Example:
```json
{
  "global": {
    "agent": {"position": [0, 0], "facing": "north", "confidence": "high"},
  },
}
```
"""


@dataclass
class CogMapMetrics:
    """Container for similarity metrics with helpers."""
    dir: float = 0.0
    facing: float = 0.0
    pos: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"dir": self.dir, "facing": self.facing, "pos": self.pos, "overall": self.overall}

    @staticmethod
    def average(items: List['CogMapMetrics']) -> 'CogMapMetrics':
        if not items:
            return CogMapMetrics()
        return CogMapMetrics(
            dir=float(np.mean([i.dir for i in items])),
            facing=float(np.mean([i.facing for i in items])),
            pos=float(np.mean([i.pos for i in items])),
            overall=float(np.mean([i.overall for i in items])),
        )


@dataclass
class CognitiveMapTurnLog:
    """Per-turn metrics for cognitive map evaluation."""
    # Hierarchical metrics
    global_metrics: CogMapMetrics = field(default_factory=CogMapMetrics)
    local_metrics: CogMapMetrics = field(default_factory=CogMapMetrics)
    rooms_metrics: CogMapMetrics = field(default_factory=CogMapMetrics)
    gates: Dict[str, float] = field(default_factory=lambda: {"conn_acc": 0.0})
    # Extraction status
    extraction_success: bool = False
    # Optional for debugging/inspection (predicted global map as BaseRoom)
    pred_room_state: Optional['BaseRoom'] = None

    # Backward-compatible flat fields reflecting global metrics
    dir_sim: float = 0.0
    facing_sim: float = 0.0
    pos_sim: float = 0.0
    overall_sim: float = 0.0

    def to_dict(self):
        return {
            "global": self.global_metrics.to_dict(),
            "local": self.local_metrics.to_dict(),
            "rooms": self.rooms_metrics.to_dict(),
            "gates": self.gates,
            "extraction_success": self.extraction_success,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {},
            # flat for backward-compatibility
            "dir_sim": self.dir_sim,
            "facing_sim": self.facing_sim,
            "pos_sim": self.pos_sim,
            "overall_sim": self.overall_sim,
        }


# =============================== Small transforms =============================== 

def rotation_matrix_from_ori(ori: np.ndarray) -> np.ndarray:
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


def transform_point(pos_world: np.ndarray, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = rotation_matrix_from_ori(anchor_ori)
    return (R @ (pos_world.astype(float) - anchor_pos.astype(float))).astype(float)


def transform_ori(ori_world: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = rotation_matrix_from_ori(anchor_ori)
    v = (R @ ori_world.astype(float)).astype(int)
    vx, vy = int(np.sign(v[0])), int(np.sign(v[1]))
    return np.array([vx, vy], dtype=int)


def _transform_baseroom(room: BaseRoom, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> BaseRoom:
    objects: List[Object] = []
    for obj in room.objects:
        p = transform_point(obj.pos, anchor_pos, anchor_ori)
        o = obj.ori
        if getattr(obj, 'has_orientation', True):
            o = transform_ori(obj.ori, anchor_ori)
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
    
    def __init__(self, cogmap_type: str = "standard", grid_size: int = 5, pos_allow_scale: bool = True):
        """Initialize cognitive map manager."""
        self.turn_logs: List[CognitiveMapTurnLog] = []
        self.cogmap_summary = copy.deepcopy(self.DEFAULT_COGMAP_SUMMARY)

        self.config = {
            "cogmap_type": cogmap_type,
            "grid_size": grid_size,
            "pos_allow_scale": bool(pos_allow_scale),
        }
        # room_id -> first-entry gate name
        self.entry_gate_by_room: dict[int, str] = {}

    def get_cognitive_map_instruction(self) -> str:
        assert self.config['cogmap_type'] == "standard", "Only standard format is supported"
        return COGMAP_INSTRUCTION_SHORTER.format(grid_size=self.config["grid_size"])
        
    def evaluate_cognitive_map(self, assistant_response: str, gt_room: Room, gt_agent: Agent) -> Optional[Dict[str, Any]]:
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

        # Parse sections
        pred_global_br, pred_local_br, pred_rooms_map, pred_gates = self._parse_predicted_maps(json_data)

        # Build GT BaseRooms
        gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent)
        gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
        gt_rooms_map = self._build_gt_room_baserooms(gt_room, gt_agent)

        # Evaluate global (used for overall summary)
        global_m = CogMapMetrics()
        if pred_global_br is not None:
            d, f, p, o = self._compare_baserooms(pred_global_br, gt_global_br)
            global_m = CogMapMetrics(d, f, p, o)

        # Evaluate local
        local_m = CogMapMetrics()
        if pred_local_br is not None:
            d, f, p, o = self._compare_baserooms(pred_local_br, gt_local_br)
            local_m = CogMapMetrics(d, f, p, o)

        # Evaluate rooms (average across rooms that exist in both)
        rooms_metrics: List[Tuple[float, float, float, float]] = []
        for rid_str, pred_br in pred_rooms_map.items():
            rid = int(rid_str)
            gt_br = gt_rooms_map.get(rid)
            if gt_br is None:
                continue
            rooms_metrics.append(self._compare_baserooms(pred_br, gt_br))
        rooms_m = CogMapMetrics.average([CogMapMetrics(*m) for m in rooms_metrics])

        # Evaluate gates connectivity
        gate_acc = self._evaluate_gate_connections(pred_gates, gt_room)

        metrics = {
            "global": global_m.to_dict(),
            "local": local_m.to_dict(),
            "rooms": rooms_m.to_dict(),
            "gates": {"conn_acc": gate_acc},
        }

        # Log all results (global fields kept for summary compatibility)
        turn_log = CognitiveMapTurnLog(
            global_metrics=global_m,
            local_metrics=local_m,
            rooms_metrics=rooms_m,
            gates={"conn_acc": gate_acc},
            extraction_success=True,
            pred_room_state=pred_global_br,
            # flat copies
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

        g_avg = CogMapMetrics.average([l.global_metrics for l in successful_logs]).to_dict()
        l_avg = CogMapMetrics.average([l.local_metrics for l in successful_logs]).to_dict()
        r_avg = CogMapMetrics.average([l.rooms_metrics for l in successful_logs]).to_dict()
        gate_avg = float(np.mean([l.gates.get("conn_acc", 0.0) for l in successful_logs]))
        
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
        patterns = [
            r'```json\s*\n(.*?)\n\s*```',
            r'```\s*\n(.*?)\n\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    json_text = match.strip()
                    json_data = json.loads(json_text)
                    if isinstance(json_data, dict) and json_data:
                        return json_data
                except (json.JSONDecodeError, ValueError):
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
                facing = obj_info.get('facing', 'north')
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
        # include agent
        objs.append(Object(name='agent', pos=gt_agent.pos.copy(), ori=gt_agent.ori.copy(), has_orientation=True))
        return BaseRoom(objects=objs, name='gt')

    def _build_gt_global_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        return _transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)

    def _build_gt_local_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        half_fov = float(getattr(DegreeRel, 'FIELD_OF_VIEW', 90.0)) / 2.0
        objs: List[Object] = [Object(name='agent', pos=np.array([0.0, 0.0]), ori=np.array([0, 1]), has_orientation=True)]
        for o in getattr(gt_room, 'objects', []):
            deg = DegreeRel.from_positions(o.pos, gt_agent.pos, gt_agent.ori).degree
            if abs(deg) <= half_fov + 1e-3:
                p = transform_point(o.pos, gt_agent.pos, gt_agent.ori)
                o_rel = transform_ori(o.ori, gt_agent.ori) if getattr(o, 'has_orientation', True) else o.ori
                objs.append(Object(name=o.name, pos=p, ori=o_rel, has_orientation=getattr(o, 'has_orientation', True)))
        return BaseRoom(objects=objs, name='gt_local')

    def _build_gt_room_baserooms(self, gt_room: Room, gt_agent: Agent) -> Dict[int, BaseRoom]:
        out: Dict[int, BaseRoom] = {}
        if not isinstance(gt_room, Room):
            return out
        for rid in sorted(getattr(gt_room, 'objects_by_room', {}).keys()):
            gate_name = self.entry_gate_by_room.get(int(rid))
            if gate_name is None:
                for g in getattr(gt_room, 'gates', []):
                    if isinstance(g.room_id, (list, tuple)) and int(rid) in [int(x) for x in g.room_id]:
                        gate_name = g.name
                        break
            if gate_name is None:
                continue
            gate = next((g for g in getattr(gt_room, 'gates', []) if g.name == gate_name), None)
            if gate is None:
                continue
            anchor_pos = gate.pos
            anchor_ori = gate.get_ori_for_room(int(rid))
            objs: List[Object] = []
            # agent in this room?
            if int(getattr(gt_agent, 'room_id', -999)) == int(rid):
                a_p = transform_point(gt_agent.pos, anchor_pos, anchor_ori)
                a_o = transform_ori(gt_agent.ori, anchor_ori)
                objs.append(Object(name='agent', pos=a_p, ori=a_o, has_orientation=True))
            # objects in this room
            for name in gt_room.objects_by_room.get(int(rid), []):
                o = gt_room.get_object_by_name(name)
                p = transform_point(o.pos, anchor_pos, anchor_ori)
                o_rel = transform_ori(o.ori, anchor_ori) if getattr(o, 'has_orientation', True) else o.ori
                objs.append(Object(name=o.name, pos=p, ori=o_rel, has_orientation=getattr(o, 'has_orientation', True)))
            out[int(rid)] = BaseRoom(objects=objs, name=f'gt_room_{rid}')
        return out

    # =============================== Room comparisons =============================== 

    def _compare_baserooms(self, pred_room: BaseRoom, gt_room: BaseRoom) -> Tuple[float, float, float, float]:
        dir_sim = self._calculate_dir_sim(pred_room, gt_room)
        facing_sim = self._calculate_facing_sim(pred_room, gt_room)
        pos_sim = self._calculate_pos_sim(pred_room, gt_room, allow_scale=bool(self.config.get('pos_allow_scale', True)))
        overall_sim = 0.5 * dir_sim + 0.2 * facing_sim + 0.3 * pos_sim
        return dir_sim, facing_sim, pos_sim, overall_sim

    def _calculate_dir_sim(self, pred_room: BaseRoom, gt_room: BaseRoom) -> float:
        """Pairwise allocentric bin agreement over shared object names."""
        pred = {o.name: o for o in pred_room.objects}
        gt = {o.name: o for o in gt_room.objects}
        names = sorted(set(pred.keys()).intersection(set(gt.keys())))
        if len(names) < 2:
            return 0.0
        bin_system = CardinalBinsAllo()
        tot = cor = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = gt[names[i]], gt[names[j]]
                gt_rel = PairwiseRelationshipDiscrete.relationship(a.pos, b.pos, None, bin_system)
                p1, p2 = pred[names[i]], pred[names[j]]
                pr = PairwiseRelationshipDiscrete.relationship(p1.pos, p2.pos, None, bin_system)
                if pr.direction.bin_id == gt_rel.direction.bin_id:
                    cor += 1.0
                tot += 1.0
        return cor / tot if tot else 0.0

    def _calculate_facing_sim(self, pred_room: BaseRoom, gt_room: BaseRoom) -> float:
        pred = {o.name: o for o in pred_room.objects}
        gt = {o.name: o for o in gt_room.objects}
        names = sorted(set(pred.keys()).intersection(set(gt.keys())))
        tot = cor = 0.0
        for name in names:
            g = gt[name]
            if not getattr(g, 'has_orientation', False):
                continue
            p = pred.get(name)
            if p is None: 
                continue
            tot += 1.0
            if np.array_equal(p.ori, g.ori): 
                cor += 1.0
        return cor / tot if tot else 0.0

    def _calculate_pos_sim(self, pred_room: BaseRoom, gt_room: BaseRoom, allow_scale: bool = True) -> float:
        """Position similarity with optional scale alignment.

        Given matched points P_pred and P_gt (same name ordering):
        - If allow_scale: find s* that minimizes ||s*·P_pred − P_gt|| in least squares
          s* = (Σ r_i·e_i) / (Σ e_i·e_i), where e_i from pred, r_i from gt
        - RMSE = sqrt(mean(||s*·e_i − r_i||^2))
        - Normalize by L = sqrt(mean(||r_i||^2)) and convert to similarity via exp(−RMSE/L)
        """
        pred = {o.name: o for o in pred_room.objects}
        gt = {o.name: o for o in gt_room.objects}
        names = sorted(set(pred.keys()).intersection(set(gt.keys())))
        if len(names) == 0:
            return 0.0
        P1 = np.array([pred[n].pos for n in names], dtype=float)
        P2 = np.array([gt[n].pos for n in names], dtype=float)
        if allow_scale:
            den = float((P1 * P1).sum())
            if den == 0.0: 
                return 0.0
            scale = float((P2 * P1).sum()) / den
        else:
            scale = 1.0
        rmse = np.sqrt(((P1 * scale - P2) ** 2).sum(axis=1).mean())
        L = np.sqrt((P2 ** 2).sum(axis=1).mean())
        return float(np.exp(-rmse / L)) if L > 0 else 0.0

    # =============================== Room entry tracking =============================== 
    def register_room_entry(self, room_id: int, gate_name: str) -> None:
        """Record the first gate used to enter a room."""
        rid = int(room_id)
        if rid not in self.entry_gate_by_room:
            self.entry_gate_by_room[rid] = gate_name

    # =============================== Gates evaluation =============================== 
    def _evaluate_gate_connections(self, pred_gates: Dict[str, Any], gt_room: Room) -> float:
        if not isinstance(gt_room, Room):
            return 0.0
        gt_map: Dict[str, List[int]] = {k: [int(x) for x in v] for k, v in getattr(gt_room, 'rooms_by_gate', {}).items()}
        if not gt_map:
            return 0.0
        correct = tot = 0
        for gate_name, gt_conn in gt_map.items():
            tot += 1
            pred = pred_gates.get(gate_name, {}) if isinstance(pred_gates, dict) else {}
            pred_conn = pred.get('connects', []) if isinstance(pred, dict) else []
            try:
                pred_conn_int = sorted([int(x) for x in pred_conn])
            except Exception:
                pred_conn_int = []
            if sorted([int(x) for x in gt_conn]) == pred_conn_int:
                correct += 1
        return float(correct) / float(tot) if tot > 0 else 0.0


if __name__ == "__main__":
    # Minimal, explicit test using the provided layout
    # Build objects (non-gates)
    objs = [
        Object('refrigerator', np.array([12, 7]), np.array([1, 0])),
        Object('chair', np.array([8, 2]), np.array([1, 0])),
        Object('bookshelf', np.array([10, 8]), np.array([0, 1])),
        Object('whiteboard', np.array([6, 6]), np.array([0, -1])),
        Object('scanner', np.array([11, 4]), np.array([1, 0])),
        Object('microwave', np.array([10, 7]), np.array([0, -1])),
        Object('monitor', np.array([13, 10]), np.array([-1, 0])),
        Object('printer', np.array([10, 2]), np.array([0, 1])),
    ]
    # Gates
    gates = [
        Gate(
            name='door_0', pos=np.array([7, 5]), ori=np.array([1, 0]),
            room_id=[2, 3], ori_by_room={2: np.array([-1, 0]), 3: np.array([1, 0])}
        ),
        Gate(
            name='door_1', pos=np.array([11, 6]), ori=np.array([0, 1]),
            room_id=[3, 1], ori_by_room={3: np.array([0, -1]), 1: np.array([0, 1])}
        ),
    ]
    # Mask (only mark required cells)
    mask = np.zeros((15, 15), dtype=np.int8)
    for (x, y) in [(12, 7), (10, 8), (10, 7), (13, 10)]:
        mask[x, y] = 1
    for (x, y) in [(6, 6)]:
        mask[x, y] = 2
    for (x, y) in [(8, 2), (11, 4), (10, 2)]:
        mask[x, y] = 3
    room = Room(objects=objs, mask=mask, name='room', gates=gates)

    agent = Agent(
        name='agent', pos=np.array([13, 9]), ori=np.array([0, 1]),
        room_id=1, init_pos=np.array([13, 9]), init_ori=np.array([0, 1]), init_room_id=1
    )

    print(room)
    print(room.gates)
    print(agent)

    # Assistant response with correct global positions (agent origin, +Y forward)
    assistant_response = (
        """```json
{
  "global": {
    "agent": {"position": [0, 0], "facing": "north", "confidence": "high"},
    "monitor": {"position": [0, 1], "facing": "west", "confidence": "high"},
    "refrigerator": {"position": [-1, -2], "facing": "east", "confidence": "high"}
  },
  "gates": {
    "door_0": {"connects": [2, 3]},
    "door_1": {"connects": [3, 1]}
  }
}
```"""
    )

    mgr = CognitiveMapManager()
    metrics = mgr.evaluate_cognitive_map(assistant_response, room, agent)
    print("metrics:", metrics)
    print("summary:", mgr.get_cogmap_summary())