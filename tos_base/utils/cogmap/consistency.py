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
    """Per-adjacent-turn update/stability metrics.

    For each adjacent exploration turn (t-1 -> t):
    - Update metric (only for observed objects in turn t):
      - position_update: Check if predicted position at turn t is getting closer to GT compared to turn t-1 (or equal)
      - facing_update: Check if facing turns from wrong to correct, or keeps unchanged
    - Stability metrics (only for unobserved objects in turn t):
      - position_stability: Check if predicted position does not get worse vs previous turn
      - facing_stability: Check if facing does not get worse vs previous turn

    Returns:
        Dict with keys: 'position_update', 'facing_update', 'position_stability', 'facing_stability'
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
        'position_stability': [],
        'facing_stability': []
    }

    if len(expl) <= 1:
        return out

    def _pos_non_worse(name: str,
                       pred_prev_dict: Dict[str, Any], pred_curr_dict: Dict[str, Any],
                       gt_prev_dict: Dict[str, Any], gt_curr_dict: Dict[str, Any]) -> float | None:
        if name not in pred_prev_dict or name not in pred_curr_dict or name not in gt_prev_dict or name not in gt_curr_dict:
            return None
        prev_dist = np.linalg.norm(np.array(pred_prev_dict[name].pos) - np.array(gt_prev_dict[name].pos))
        curr_dist = np.linalg.norm(np.array(pred_curr_dict[name].pos) - np.array(gt_curr_dict[name].pos))
        return 1.0 if curr_dist <= prev_dist else 0.0

    def _facing_non_worse(name: str,
                          pred_prev_dict: Dict[str, Any], pred_curr_dict: Dict[str, Any],
                          gt_prev_dict: Dict[str, Any], gt_curr_dict: Dict[str, Any]) -> float | None:
        if not _is_valid_for_facing(name, gt_curr_dict, gt_prev_dict):
            return None
        if name not in pred_prev_dict or name not in pred_curr_dict or name not in gt_prev_dict or name not in gt_curr_dict:
            return None
        prev_correct = np.array_equal(pred_prev_dict[name].ori, gt_prev_dict[name].ori)
        curr_correct = np.array_equal(pred_curr_dict[name].ori, gt_curr_dict[name].ori)
        if prev_correct and not curr_correct:
            return 0.0
        if curr_correct and not prev_correct:
            return 1.0
        unchanged = np.array_equal(pred_curr_dict[name].ori, pred_prev_dict[name].ori)
        return 1.0 if unchanged else 0.0

    for i in range(1, len(expl)):
        prev_log = expl[i - 1]
        curr_log = expl[i]
        prev_exp = prev_log.get('exploration_log') or {}
        curr_exp = curr_log.get('exploration_log') or {}
        
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
        # Calculate only for observed objects
        pos_update_scores: List[float] = []
        facing_update_scores: List[float] = []
        
        if observed_set:
            for name in observed_set:
                if name == 'agent':
                    continue
                s = _pos_non_worse(name, pred_prev_dict, pred_curr_dict, gt_prev_dict, gt_curr_dict)
                if s is not None:
                    pos_update_scores.append(s)

                s = _facing_non_worse(name, pred_prev_dict, pred_curr_dict, gt_prev_dict, gt_curr_dict)
                if s is not None:
                    facing_update_scores.append(s)

        out['position_update'].append(float(np.mean(pos_update_scores)) if pos_update_scores else None)
        out['facing_update'].append(float(np.mean(facing_update_scores)) if facing_update_scores else None)

        # --- Stability Metrics (Unobserved objects) ---
        common_names = set(pred_curr_dict) & set(pred_prev_dict) & set(gt_curr_dict) & set(gt_prev_dict)
        unobserved = common_names - observed_set - {'agent'}

        pos_stab_scores: List[float] = []
        fac_stab_scores: List[float] = []
        for name in unobserved:
            s = _pos_non_worse(name, pred_prev_dict, pred_curr_dict, gt_prev_dict, gt_curr_dict)
            if s is not None:
                pos_stab_scores.append(s)

            s = _facing_non_worse(name, pred_prev_dict, pred_curr_dict, gt_prev_dict, gt_curr_dict)
            if s is not None:
                fac_stab_scores.append(s)

        out['position_stability'].append(float(np.mean(pos_stab_scores)) if pos_stab_scores else None)
        out['facing_stability'].append(float(np.mean(fac_stab_scores)) if fac_stab_scores else None)

    return out


__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "stability",
]

if __name__ == "__main__":
    pass
