from typing import Dict, List, Any, Tuple
import numpy as np

def calculate_confidence_metrics(env_data: Dict[str, Any], threshold: int = 2) -> Tuple[List[float], List[float]]:
    """
    Calculate confidence match and confidence ratio metrics per turn.
    Confidence match: if the max distance between possible positions > threshold (default 2),
    then confidence should be low; otherwise high.
    Confidence ratio: the ratio of high confidence objects to total objects.
    
    Args:
        env_data: Dictionary containing environment logs
        threshold: Threshold for max position spread to determine low/high confidence (default 2)
        
    Returns:
        Tuple of (confidence_match_per_turn, confidence_ratio_per_turn)
    """
    turn_logs = env_data.get('env_turn_logs') or []
    conf_match_list = []
    conf_ratio_list = []

    def _max_pairwise_dist(pts: List[List[int]] | None) -> float:
        if not pts or len(pts) < 2:
            return 0.0
        P = np.asarray(pts, dtype=float)
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        return float(np.sqrt(d2.max()))
    
    for log in turn_logs:
        # Only process exploration turns with cogmap logs
        if not log.get('is_exploration_phase') or not log.get('cogmap_log'):
            continue
            
        cog_log = log.get('cogmap_log', {})
        global_log = cog_log.get('global', {})
        if not global_log.get('extraction_success'):
            conf_match_list.append(0.0)
            conf_ratio_list.append(0.0)
            continue
            
        pred_json = global_log.get('pred_json', {})
        possible_positions = (log.get('exploration_log') or {}).get('possible_positions') or {}
        
        match_scores = []
        high_conf_count = 0
        total_objects = 0
        
        for obj_name, obj_data in pred_json.items():
            if not isinstance(obj_data, dict):
                continue

            conf_str = str(obj_data.get('confidence', '')).lower()
            if conf_str not in ('high', 'low'):
                continue
            pts = possible_positions.get(obj_name)
            if pts is None:
                pts = possible_positions.get(obj_name.replace('_', ' '))
            if pts is None:
                continue

            max_dist = _max_pairwise_dist(pts)
            is_match = (max_dist > threshold and conf_str == 'low') or (max_dist <= threshold and conf_str == 'high')
            
            match_scores.append(1.0 if is_match else 0.0)
            
            # Confidence Ratio
            if conf_str == 'high':
                high_conf_count += 1
            total_objects += 1
            
        avg_match = float(np.mean(match_scores)) if match_scores else 0.0
        conf_match_list.append(avg_match)
        
        ratio = float(high_conf_count / total_objects) if total_objects > 0 else 0.0
        conf_ratio_list.append(ratio)
        
    return conf_match_list, conf_ratio_list
