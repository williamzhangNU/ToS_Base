from typing import Dict, Tuple, List, Any
import numpy as np
import copy

from ...core.room import BaseRoom, Room, Object
from ...core.object import Agent 
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

def _is_valid_for_facing(name: str, gt_curr_dict: Dict, gt_prev_dict: Dict) -> bool:
    """Check if object is valid for facing evaluation (not agent, not gate, has orientation)."""
    if name == 'agent':
        return False
        
    # Check exclusion based on GT if available
    if name in gt_curr_dict:
        gt_obj = gt_curr_dict[name]
        is_gate = "door" in name.lower() or "gate" in name.lower()
        if not gt_obj.has_orientation or is_gate:
            return False
    elif name in gt_prev_dict:
        gt_obj = gt_prev_dict[name]
        is_gate = "door" in name.lower() or "gate" in name.lower()
        if not gt_obj.has_orientation or is_gate:
            return False
    else:
        # Fallback exclusion based on name
        if "door" in name.lower() or "gate" in name.lower():
            return False
            
    return True

def stability(env_data_or_logs: Dict | List[Dict], threshold: int = 1,
              allow_scale: bool = False, pos_norm_L: float | None = None) -> Dict[str, List[float]]:
    """Per-adjacent-turn stability decoupled into update and stability check metrics.

    For each adjacent exploration turn (t-1 -> t):
    - Update metric (only for observed objects):
      - position_update: Check if predicted position at turn t is getting closer to GT compared to turn t-1 (or equal)
      - facing_update: Check if facing turns from wrong to correct, or keeps unchanged
    - Stability check metric (for objects with small domain-size change):
      - stability: 0.5 * pos_acc + 0.5 * dir_acc
    - Facing stability metric (for unobserved objects):
      - facing_stability: Check if facing matches previous turn

    Returns:
        Dict with keys: 'position_update', 'facing_update', 'stability', 'facing_stability'
        Each value is a List[float] (or None where invalid/not applicable)
    """
    # Normalize input to a list of exploration turns
    if isinstance(env_data_or_logs, dict):
        logs = env_data_or_logs.get('env_turn_logs', []) or []
    else:
        logs = env_data_or_logs or []

    expl = [t for t in logs if t.get('is_exploration_phase')]
    
    out = {
        'position_update': [],
        'facing_update': [],
        'stability': [],
        'facing_stability': []
    }

    if len(expl) <= 1:
        return out

    def _filter_room(br: BaseRoom, keep: set[str]) -> BaseRoom:
        objs = [o for o in br.objects if o.name in keep]
        return BaseRoom(objects=objs, name=br.name)

    for i in range(1, len(expl)):
        prev_log = expl[i - 1]
        curr_log = expl[i]
        prev_exp = prev_log.get('exploration_log') or {}
        curr_exp = curr_log.get('exploration_log') or {}
        
        # Possible positions for stability check
        prev_pp: Dict[str, List[List[int]]] = prev_exp.get('possible_positions') or {}
        curr_pp: Dict[str, List[List[int]]] = curr_exp.get('possible_positions') or {}

        # Observed objects in *this* turn:
        # - Prefer `visible_objects` (per-turn).
        # - Fallback: derive newly-observed items from cumulative `observed_items`.
        observed_set = set(curr_exp.get('visible_objects') or [])

        # Need previous and current predicted and GT global rooms
        g_prev = ((prev_log.get('cogmap_log') or {}).get('global') or {})
        pred_prev = BaseRoom.from_dict((g_prev.get('pred_room_state')) or {})
        gt_prev = BaseRoom.from_dict((g_prev.get('gt_room_state_full') or g_prev.get('gt_room_state')) or {})

        g_curr = ((curr_log.get('cogmap_log') or {}).get('global') or {})
        pred_curr = BaseRoom.from_dict((g_curr.get('pred_room_state')) or {})
        gt_curr = BaseRoom.from_dict((g_curr.get('gt_room_state_full') or g_curr.get('gt_room_state')) or {})

        if pred_prev is None or pred_curr is None or gt_prev is None or gt_curr is None:
            out['position_update'].append(None)
            out['facing_update'].append(None)
            out['stability'].append(None)
            out['facing_stability'].append(None)
            continue

        pred_prev_dict = {o.name: o for o in pred_prev.objects}
        pred_curr_dict = {o.name: o for o in pred_curr.objects}
        gt_prev_dict = {o.name: o for o in gt_prev.objects}
        gt_curr_dict = {o.name: o for o in gt_curr.objects}

        # --- Update Metrics (Position & Facing) ---
        # Calculate for observed objects
        pos_update_scores: List[float] = []
        facing_update_scores: List[float] = []
        
        if observed_set:
            for name in observed_set:
                if name == 'agent':
                    continue
                
                if name in pred_prev_dict and name in pred_curr_dict and name in gt_prev_dict and name in gt_curr_dict:
                    # Position Update
                    prev_dist = np.linalg.norm(np.array(pred_prev_dict[name].pos) - np.array(gt_prev_dict[name].pos))
                    curr_dist = np.linalg.norm(np.array(pred_curr_dict[name].pos) - np.array(gt_curr_dict[name].pos))
                    pos_update_scores.append(1.0 if curr_dist <= prev_dist else 0.0)
                    
                    # Facing Update
                    if _is_valid_for_facing(name, gt_curr_dict, gt_prev_dict):
                        curr_correct = np.array_equal(pred_curr_dict[name].ori, gt_curr_dict[name].ori)
                        prev_correct = np.array_equal(pred_prev_dict[name].ori, gt_prev_dict[name].ori)
                        unchanged = np.array_equal(pred_curr_dict[name].ori, pred_prev_dict[name].ori)
                        
                        if (curr_correct and not prev_correct) or unchanged:
                            facing_update_scores.append(1.0)
                        else:
                            facing_update_scores.append(0.0)

        out['position_update'].append(float(np.mean(pos_update_scores)) if pos_update_scores else None)
        out['facing_update'].append(float(np.mean(facing_update_scores)) if facing_update_scores else None)

        # --- Stability Check Metric ---
        # Condition: abs(len(prev_pts) - len(curr_pp[name])) < threshold (1)
        stability_score = None
        if prev_pp and curr_pp:
            selected_stability: set[str] = set()
            for name, prev_pts in prev_pp.items():
                if name == 'agent':
                    continue
                if name in curr_pp:
                    if abs(len(prev_pts) - len(curr_pp[name])) < int(threshold):
                        selected_stability.add(name)
            
            if selected_stability:
                pred_sel = _filter_room(pred_curr, selected_stability)
                gt_sel = _filter_room(gt_curr, selected_stability)
                metrics = compare_on_common_subset(pred_sel, gt_sel, allow_scale=allow_scale, pos_norm_L=pos_norm_L)
                if metrics.valid:
                    stability_score = 0.5 * metrics.pos + 0.5 * metrics.dir
        
        out['stability'].append(stability_score)

        # --- Facing Stability Metric ---
        # Only focus on facing. Calculated when in current turn, if object is NOT observed.
        # If facing matches previous turn -> 1, else 0.
        facing_stability_scores: List[float] = []
        
        # Check all objects present in both predictions that are NOT in observed_set
        common_pred_names = set(pred_curr_dict.keys()) & set(pred_prev_dict.keys())
        unobserved_in_pred = common_pred_names - observed_set
        
        if unobserved_in_pred:
            for name in unobserved_in_pred:
                if not _is_valid_for_facing(name, gt_curr_dict, gt_prev_dict):
                    continue

                match = np.array_equal(pred_curr_dict[name].ori, pred_prev_dict[name].ori)
                facing_stability_scores.append(1.0 if match else 0.0)
        
        out['facing_stability'].append(float(np.mean(facing_stability_scores)) if facing_stability_scores else None)

    return out


__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "stability",
]

if __name__ == "__main__":
    pass
