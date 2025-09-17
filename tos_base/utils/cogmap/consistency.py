from typing import Dict, Tuple, List
import numpy as np

from ...core.room import BaseRoom, Room
from ...core.object import Agent
from .transforms import br_from_anchor_to_initial, transform_baseroom
from .metrics import compute_map_metrics
from .types import MapCogMetrics, ConsistencySummary


def compare_on_common_subset(a: BaseRoom | None, b: BaseRoom | None, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if a is None or b is None:
        return MapCogMetrics.invalid()
    names_a = {o.name for o in a.objects}
    names_b = {o.name for o in b.objects}
    names = names_a & names_b
    if not names:
        return MapCogMetrics.invalid()
    a_sub = BaseRoom(objects=[o for o in a.objects if o.name in names], name=a.name)
    b_sub = BaseRoom(objects=[o for o in b.objects if o.name in names], name=b.name)
    return compute_map_metrics(a_sub, b_sub, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def local_vs_global_consistency(pred_local: BaseRoom | None, pred_global: BaseRoom | None, agent: Agent, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if pred_local is None or pred_global is None:
        return MapCogMetrics.invalid()
    local_in_initial = br_from_anchor_to_initial(pred_local, np.array(agent.pos, dtype=float), np.array(agent.ori, dtype=int), agent)
    return compare_on_common_subset(local_in_initial, pred_global, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def rooms_vs_global_consistency(pred_rooms: Dict[str, BaseRoom], pred_global: BaseRoom | None, room: Room, agent: Agent, entry_gate_by_room: Dict[int, str], allow_scale: bool, pos_norm_L: float | None) -> Tuple[MapCogMetrics, Dict[str, MapCogMetrics]]:
    if pred_global is None:
        return MapCogMetrics.invalid(), {}
    per_room: Dict[str, MapCogMetrics] = {}
    vals: List[MapCogMetrics] = []
    for rid, room_br in sorted(pred_rooms.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]):
        gate_name = entry_gate_by_room.get(int(rid))
        if not gate_name:
            continue
        g = next((gg for gg in room.gates if gg.name == gate_name), None)
        if g is None:
            continue
        gate_pos = g.pos
        gate_ori = g.get_ori_for_room(int(rid))
        room_in_initial = br_from_anchor_to_initial(room_br, gate_pos, gate_ori, agent)
        m = compare_on_common_subset(room_in_initial, pred_global, allow_scale=allow_scale, pos_norm_L=pos_norm_L)
        if m.valid:
            per_room[rid] = m
            vals.append(m)
    avg = MapCogMetrics.average(vals) if vals else MapCogMetrics.invalid()
    return avg, per_room


def map_vs_relations_consistency(_: Dict, __: Dict) -> float:
    return 0.0


def relations_consistency(_: Dict) -> float:
    return 0.0


__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "rooms_vs_global_consistency",
    "map_vs_relations_consistency",
    "relations_consistency",
]


