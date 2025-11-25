from typing import Dict, List, Any, Tuple
import numpy as np

def calculate_confidence_metrics(env_data: Dict[str, Any], threshold: int = 2) -> Tuple[List[float], List[float]]:
    """
    Calculate confidence match and confidence ratio metrics per turn.
    
    Args:
        env_data: Dictionary containing environment logs
        threshold: Threshold for number of possible positions to determine low/high confidence (default 2)
        
    Returns:
        Tuple of (confidence_match_per_turn, confidence_ratio_per_turn)
    """
    turn_logs = env_data.get('env_turn_logs') or []
    conf_match_list = []
    conf_ratio_list = []
    
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
        
        # Calculate Confidence Match
        # If number of possible positions > threshold (default 2), then confidence should be low, otherwise not match (high).
        match_scores = []
        high_conf_count = 0
        total_objects = 0
        
        for obj_name, obj_data in pred_json.items():
            if not isinstance(obj_data, dict):
                continue
                
            # Skip agent if needed, but requirements say "all objects in the global map"
            # Usually agent is in pred_json, let's include it as it has confidence in the example
            
            conf_str = str(obj_data.get('confidence', '')).lower()
            if conf_str not in ['high', 'low']:
                # If confidence is missing or invalid, treat as mismatch? Or skip?
                # Let's treat as mismatch (0.0) if we can't determine
                # But if it's missing, maybe we shouldn't count it?
                # Requirement: "calculate the average of confidence matches of all objects"
                # If confidence is missing, it's a failure to follow protocol.
                pass 

            # Get number of possible positions
            # If object not in possible_positions, what to do?
            # possible_positions tracks objects. If it's a new object, maybe we don't know?
            # Assuming possible_positions contains all tracked objects.
            # If not found, maybe default to 0 or 1? 
            # If we don't know the possible positions, we can't evaluate match.
            # Let's assume if it's in pred_json, we should try to evaluate it.
            # If not in possible_positions, maybe it's fully determined (1 pos)? Or unknown?
            # Let's use the list if available.
            
            num_pos = len(possible_positions.get(obj_name, []))
            
            # Logic:
            # > threshold -> expected low
            # <= threshold -> expected high (implied "otherwise not match" logic)
            
            is_match = (num_pos > threshold and conf_str == 'low') or \
                (num_pos <= threshold and conf_str == 'high')
            
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
