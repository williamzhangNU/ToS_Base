from typing import Dict, Tuple, List
import numpy as np
import copy

from ...core.room import BaseRoom, Room, Object
from ...core.object import Agent 
from .transforms import br_from_anchor_to_initial
from .metrics import compute_map_metrics
from .types import MapCogMetrics
from .transforms import transform_baseroom


def compare_on_common_subset(a: BaseRoom | None, b: BaseRoom | None, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if a is None or b is None:
        return MapCogMetrics.invalid()
    names_a = {o.name for o in a.objects}
    if not names_a:
        return MapCogMetrics(dir=1.0, facing=1.0, overall=1.0, pos=1.0)  # No objects in A, trivially perfect
    names_b = {o.name for o in b.objects}
    names = names_a & names_b
    if not names:
        # No overlap -> treat as wrong
        return MapCogMetrics(dir=0.0, facing=0.0, overall=0.0, pos=0.0, valid=True)
    a_sub = BaseRoom(objects=[o for o in a.objects if o.name in names], name=a.name)
    b_sub = BaseRoom(objects=[o for o in b.objects if o.name in names], name=b.name)
    return compute_map_metrics(a_sub, b_sub, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def local_vs_global_consistency(pred_local: BaseRoom | None, pred_global: BaseRoom | None, agent: Agent, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if pred_local is None or pred_global is None:
        return MapCogMetrics.invalid()
    
    # Find predicted agent in global map
    global_agent = next((o for o in pred_global.objects if o.name == 'agent'), None)
    if global_agent is None:
        return MapCogMetrics.invalid()
    
    # Transform global map to use predicted agent as origin (make copy to avoid modifying original)
    
    global_copy = copy.deepcopy(pred_global)
    global_agent_centered = transform_baseroom(global_copy, global_agent.pos, global_agent.ori)
    
    # Compare directly (local should already be agent-centered)
    return compare_on_common_subset(pred_local, global_agent_centered, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def rooms_vs_global_consistency(pred_rooms: Dict[str, BaseRoom], pred_global: BaseRoom | None, room: Room, agent: Agent, entry_gate_by_room: Dict[int, str], allow_scale: bool, pos_norm_L: float | None) -> Tuple[MapCogMetrics, Dict[str, MapCogMetrics]]:
    if pred_global is None:
        return MapCogMetrics.invalid(), {}
    per_room: Dict[str, MapCogMetrics] = {}
    vals: List[MapCogMetrics] = []
    # Iterate over all GT rooms to ensure missing predictions count as 0
    for rid_int in sorted(room.objects_by_room.keys()):
        rid = str(rid_int)
        room_br = pred_rooms.get(rid)
        gate_name = entry_gate_by_room.get(rid_int)
        if gate_name:
            g = next((gg for gg in room.gates if gg.name == gate_name), None)
            if g is None:
                m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
                per_room[rid] = m
                vals.append(m)
                continue
            gate_pos = g.pos
            gate_ori = g.get_ori_for_room(rid_int)
        elif rid_int == 1:
            gate_pos = agent.init_pos
            gate_ori = agent.init_ori
        else:
            # No anchor info; treat as wrong
            m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
            per_room[rid] = m
            vals.append(m)
            continue
        if room_br is None:
            m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
            per_room[rid] = m
            vals.append(m)
            continue
        room_in_initial = br_from_anchor_to_initial(room_br, gate_pos, gate_ori, agent)
        m = compare_on_common_subset(room_in_initial, pred_global, allow_scale=allow_scale, pos_norm_L=pos_norm_L)
        # If invalid comparison (no overlap), count as 0 instead of skipping
        if not m.valid:
            m = MapCogMetrics(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=True)
        per_room[rid] = m
        vals.append(m)
    avg = MapCogMetrics.average(vals) if vals else MapCogMetrics.invalid()
    return avg, per_room


def stability(env_data_or_logs: Dict | List[Dict], threshold: int = 5,
              allow_scale: bool = False, pos_norm_L: float | None = None) -> Tuple[List[float], List[MapCogMetrics]]:
    """Per-adjacent-turn stability decoupled into update and stability check metrics.

    For each adjacent exploration turn (t-1 -> t):
    - Update metric: For objects in previous turn's observed_items (re-observed objects),
      check if predicted position at turn t is getting closer to GT compared to turn t-1
    - Stability check metric: For objects with small domain-size change (based on possible_positions),
      compare current predicted global map vs current GT global

    Returns:
        Tuple of (update_metrics, stability_check_metrics):
        - update_metrics: List[float] - Average of boolean values indicating if each re-observed object is getting closer to GT
        - stability_check_metrics: List[MapCogMetrics] - Stability comparison metrics for unchanged objects
    """
    # Normalize input to a list of exploration turns
    if isinstance(env_data_or_logs, dict):
        logs = env_data_or_logs.get('env_turn_logs', []) or []
    else:
        logs = env_data_or_logs or []

    expl = [t for t in logs if t.get('is_exploration_phase')]
    update_out: List[float] = []
    stability_out: List[MapCogMetrics] = []
    if len(expl) <= 1:
        return update_out, stability_out

    def _filter_room(br: BaseRoom, keep: set[str]) -> BaseRoom:
        objs = [o for o in br.objects if o.name in keep]
        return BaseRoom(objects=objs, name=br.name)

    for i in range(1, len(expl)):
        prev_log = expl[i - 1]
        curr_log = expl[i]
        prev_pp: Dict[str, List[List[int]]] = (prev_log.get('exploration_log') or {}).get('possible_positions') or {}
        curr_pp: Dict[str, List[List[int]]] = (curr_log.get('exploration_log') or {}).get('possible_positions') or {}

        # Get observed items from previous turn's exploration log (re-observed objects)
        prev_observed: List[str] = (prev_log.get('exploration_log') or {}).get('observed_items') or []
        observed_set = set(prev_observed)

        # Need previous and current predicted and GT global rooms
        g_prev = ((prev_log.get('cogmap_log') or {}).get('global') or {})
        pred_prev = BaseRoom.from_dict((g_prev.get('pred_room_state')) or {})
        gt_prev = BaseRoom.from_dict((g_prev.get('gt_room_state_full') or g_prev.get('gt_room_state')) or {})

        g_curr = ((curr_log.get('cogmap_log') or {}).get('global') or {})
        pred_curr = BaseRoom.from_dict((g_curr.get('pred_room_state')) or {})
        gt_curr = BaseRoom.from_dict((g_curr.get('gt_room_state_full') or g_curr.get('gt_room_state')) or {})

        if pred_prev is None or pred_curr is None or gt_prev is None or gt_curr is None:
            update_out.append(0.0)
            stability_out.append(MapCogMetrics.invalid())
            continue

        # Compute update metric: check if each observed object is getting closer to GT
        update_scores: List[bool] = []
        pred_prev_dict = {o.name: o for o in pred_prev.objects}
        pred_curr_dict = {o.name: o for o in pred_curr.objects}
        gt_prev_dict = {o.name: o for o in gt_prev.objects}
        gt_curr_dict = {o.name: o for o in gt_curr.objects}

        for name in observed_set:
            # Check if object exists in all required maps
            if name in pred_prev_dict and name in pred_curr_dict and name in gt_prev_dict and name in gt_curr_dict:
                # Calculate distances to ground truth
                prev_dist = np.linalg.norm(np.array(pred_prev_dict[name].pos) - np.array(gt_prev_dict[name].pos))
                curr_dist = np.linalg.norm(np.array(pred_curr_dict[name].pos) - np.array(gt_curr_dict[name].pos))
                # Object is updating towards GT if current distance is smaller
                update_scores.append(curr_dist <= prev_dist)

        # Average of boolean values (True=1, False=0)
        update_metric = float(np.mean(update_scores)) if update_scores else 0.0
        update_out.append(update_metric)

        # Compute stability check metric: select unchanged objects based on domain-size change
        if not prev_pp or not curr_pp:
            stability_out.append(MapCogMetrics.invalid())
            continue

        selected: set[str] = set()
        for name, prev_pts in prev_pp.items():
            if abs(len(prev_pts) - len(curr_pp[name])) < int(threshold):
                selected.add(name)

        if not selected:
            stability_out.append(MapCogMetrics.invalid())
            continue

        # Existing comparison logic for stability check
        pred_sel = _filter_room(pred_curr, selected)
        gt_sel = _filter_room(gt_curr, selected)
        stability_out.append(compare_on_common_subset(pred_sel, gt_sel, allow_scale=allow_scale, pos_norm_L=pos_norm_L))

    return update_out, stability_out


__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "rooms_vs_global_consistency",
    "map_vs_relations_consistency",
    "relations_consistency",
    "stability",
]



if __name__ == "__main__":
    print("Testing consistency functions...")

    # Test 1: compare_on_common_subset
    print("\n1. Testing compare_on_common_subset:")
    try:
        # Create test BaseRooms with common objects
        obj1_a = Object(name="chair", pos=[1, 2])
        obj2_a = Object(name="table", pos=[3, 4])
        room_a = BaseRoom(objects=[obj1_a, obj2_a], name="room_a")

        obj1_b = Object(name="chair", pos=[1.1, 2.1])  # Slightly different position
        obj2_b = Object(name="table", pos=[3.2, 4.1])
        room_b = BaseRoom(objects=[obj1_b, obj2_b], name="room_b")

        metrics = compare_on_common_subset(room_a, room_b, allow_scale=False, pos_norm_L=None)
        print(f"Metrics: overall={metrics.overall:.3f}, pos={metrics.pos:.3f}, valid={metrics.valid}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: local_vs_global_consistency
    print("\n2. Testing local_vs_global_consistency:")
    try:
        # Create local and global rooms
        local_obj = Object(name="chair", pos=[0, 1])  # Relative to agent
        pred_local = BaseRoom(objects=[local_obj], name="local")

        global_obj = Object(name="chair", pos=[2, 3])  # Global position
        pred_global = BaseRoom(objects=[global_obj], name="global")

        agent = Agent(pos=[2, 2], ori=[0, 1])  # Agent at (2,2) facing north

        metrics = local_vs_global_consistency(pred_local, pred_global, agent, allow_scale=False, pos_norm_L=None)
        print(f"Metrics: overall={metrics.overall:.3f}, pos={metrics.pos:.3f}, valid={metrics.valid}")
    except Exception as e:
        print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}")

    print("\nConsistency function tests completed!")


