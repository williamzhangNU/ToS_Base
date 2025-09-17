from typing import List, Dict, Any, Callable, Optional

from .types import MapCogMetrics, RelationMetrics


def _avg(values: List[float]) -> float:
    v = [x for x in values if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else 0.0


def _avg_metrics(keys: List[str], metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {k: 0.0 for k in keys}
    return {k: _avg([m.get(k) for m in metrics_list if isinstance(m, dict)]) for k in keys}


def _avg_list_of_lists(list_of_lists: List[List[float]]) -> List[float]:
    if not list_of_lists:
        return []
    max_len = max(len(lst) for lst in list_of_lists)
    out = []
    for i in range(max_len):
        vals = [lst[i] for lst in list_of_lists if i < len(lst) and isinstance(lst[i], (int, float))]
        out.append(_avg(vals) if vals else 0.0)
    return out


def aggregate_per_sample_then_group(samples: List[Any], per_sample_fn: Callable[[Any], Dict[str, float]]) -> Dict[str, float]:
    metrics_per_sample = []
    for s in samples:
        m = per_sample_fn(s)
        if isinstance(m, dict):
            metrics_per_sample.append(m)
    # keys come from first dict or default map keys
    keys = list(metrics_per_sample[0].keys()) if metrics_per_sample else ["dir", "facing", "pos", "overall"]
    return _avg_metrics(keys, metrics_per_sample)


def aggregate_lists_per_turn(samples: List[Any], per_sample_list_fn: Callable[[Any], List[float]]) -> List[float]:
    lists = []
    for s in samples:
        lst = per_sample_list_fn(s)
        if isinstance(lst, list):
            lists.append(lst)
    return _avg_list_of_lists(lists)


def calculate_cogmap_per_turn(env_data_list: List[Dict[str, Any]], mode: str = "update") -> Dict[str, List[float]]:
    """Average global cognitive map metrics for each turn across samples.

    Returns dict with keys 'dir', 'facing', 'pos', 'overall', each a list over turns.
    """
    from collections import defaultdict

    turn_to_metrics: Dict[int, List[MapCogMetrics]] = defaultdict(list)
    for env_data in env_data_list:
        for turn_idx, turn_log in enumerate(env_data.get('env_turn_logs', [])):
            cogmap_agg = turn_log.get('cogmap_log') or {}
            level_data = cogmap_agg.get('global') or {}
            if not isinstance(level_data, dict):
                continue
            metrics_data = level_data.get('metrics_full' if mode == 'full' else 'metrics') or {}
            if not isinstance(metrics_data, dict):
                continue
            m = MapCogMetrics(
                dir=float(metrics_data.get('dir', 0.0)),
                facing=float(metrics_data.get('facing', 0.0)),
                pos=float(metrics_data.get('pos', 0.0)),
                overall=float(metrics_data.get('overall', 0.0)),
                valid=True,
            )
            turn_to_metrics[turn_idx].append(m)

    max_turn = max(turn_to_metrics.keys()) if turn_to_metrics else -1
    per_turn_avg = [MapCogMetrics.average(turn_to_metrics[i]) if i in turn_to_metrics else MapCogMetrics.invalid() for i in range(max_turn + 1)]

    return {
        'dir': [float(m.dir) if m.valid else 0.0 for m in per_turn_avg],
        'facing': [float(m.facing) if m.valid else 0.0 for m in per_turn_avg],
        'pos': [float(m.pos) if m.valid else 0.0 for m in per_turn_avg],
        'overall': [float(m.overall) if m.valid else 0.0 for m in per_turn_avg],
    }


def compute_error_aggregates(env_data_list: List[Dict[str, Any]],) -> Dict[str, Any]:
    map_keys = ["dir", "facing", "pos", "overall"]

    def _per_sample_local_err(env_data: Dict[str, Any]) -> Dict[str, float]:
        vals = []
        for t in env_data.get('env_turn_logs', []):
            m = (t.get('cogmap_log', {}) or {}).get('local', {}).get('metrics', {})
            if m:
                vals.append({k: float(m.get(k, 0.0)) for k in map_keys})
        return _avg_metrics(map_keys, vals)

    def _per_sample_global_err(env_data: Dict[str, Any]) -> Dict[str, float]:
        vals = []
        for t in env_data.get('env_turn_logs', []):
            m = (t.get('cogmap_log', {}) or {}).get('global', {}).get('metrics', {})
            if m:
                vals.append({k: float(m.get(k, 0.0)) for k in map_keys})
        return _avg_metrics(map_keys, vals)

    return {
        'local_vs_gt_local_avg': aggregate_per_sample_then_group(env_data_list, _per_sample_local_err),
        'global_vs_gt_global_avg': aggregate_per_sample_then_group(env_data_list, _per_sample_global_err),
    }


def compute_correctness_aggregates(env_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    map_keys = ["dir", "facing", "pos", "overall"]

    def _last_explore(env_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for t in reversed(env_data.get('env_turn_logs', [])):
            if t.get('cogmap_log'):
                return t.get('cogmap_log')
        return None

    def _last_global_full(env_data: Dict[str, Any]) -> Dict[str, float]:
        lg = _last_explore(env_data)
        m = (lg or {}).get('global', {}).get('metrics_full', {})
        return {k: float(m.get(k, 0.0)) for k in map_keys}

    def _last_relations_full(env_data: Dict[str, Any]) -> Dict[str, float]:
        lg = _last_explore(env_data)
        m = (lg or {}).get('relations', {}).get('metrics_full', {})
        return {k: float(m.get(k, 0.0)) for k in ['dir', 'dist', 'overall']}

    # Passive global map (full) aggregated
    passive_vals = []
    for env_data in env_data_list:
        cfg = (env_data.get('env_info') or {}).get('config') or {}
        if cfg.get('exp_type') == 'passive':
            turns = env_data.get('env_turn_logs') or []
            if turns:
                mm = (turns[0].get('cogmap_log') or {}).get('global', {}).get('metrics_full', {})
                if mm:
                    passive_vals.append({k: float(mm.get(k, 0.0)) for k in map_keys})

    return {
        'last_global_vs_gt_full': aggregate_per_sample_then_group(env_data_list, _last_global_full),
        'last_relations_vs_gt_full': aggregate_per_sample_then_group(env_data_list, _last_relations_full),
        'per_turn_global_full': calculate_cogmap_per_turn(env_data_list, mode='full'),
        'per_turn_global_observed': calculate_cogmap_per_turn(env_data_list, mode='update'),
        'passive_global_full': ( {k: (sum([d.get(k, 0.0) for d in passive_vals]) / len(passive_vals) if passive_vals else 0.0) for k in map_keys} ),
    }


def compute_consistency_aggregates(env_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    map_keys = ["dir", "facing", "pos", "overall"]

    def _avg_metrics_dict(dcts: List[Dict[str, float]]) -> Dict[str, float]:
        return {k: (sum([d.get(k, 0.0) for d in dcts]) / len(dcts) if dcts else 0.0) for k in map_keys}

    def _per_sample_local_vs_global(env_data: Dict[str, Any]) -> Dict[str, float]:
        vals = []
        for t in env_data.get('env_turn_logs', []):
            cons = (t.get('cogmap_log') or {}).get('consistency') or {}
            m = cons.get('local_vs_global') or {}
            if m:
                vals.append({k: float(m.get(k, 0.0)) for k in map_keys})
        return _avg_metrics_dict(vals)

    def _last_rooms_vs_global(env_data: Dict[str, Any]) -> Dict[str, float]:
        lg = None
        for t in reversed(env_data.get('env_turn_logs', [])):
            if t.get('cogmap_log'):
                lg = t.get('cogmap_log')
                break
        m = ((lg or {}).get('consistency', {}).get('rooms_vs_global', {}).get('average', {}))
        return {k: float(m.get(k, 0.0)) for k in map_keys}

    def _last_map_vs_rel(env_data: Dict[str, Any]) -> float:
        lg = None
        for t in reversed(env_data.get('env_turn_logs', [])):
            if t.get('cogmap_log'):
                lg = t.get('cogmap_log')
                break
        return float((lg or {}).get('consistency', {}).get('map_vs_relations', 0.0))

    def _last_rel_cons(env_data: Dict[str, Any]) -> float:
        lg = None
        for t in reversed(env_data.get('env_turn_logs', [])):
            if t.get('cogmap_log'):
                lg = t.get('cogmap_log')
                break
        return float((lg or {}).get('consistency', {}).get('relations_consistency', 0.0))

    return {
        'local_vs_global_avg': aggregate_per_sample_then_group(env_data_list, _per_sample_local_vs_global),
        'rooms_vs_global_last': aggregate_per_sample_then_group(env_data_list, _last_rooms_vs_global),
        'map_vs_relations_last': (sum([_last_map_vs_rel(s) for s in env_data_list]) / len(env_data_list)) if env_data_list else 0.0,
        'relations_consistency_last': (sum([_last_rel_cons(s) for s in env_data_list]) / len(env_data_list)) if env_data_list else 0.0,
    }


__all__ = [
    "aggregate_per_sample_then_group",
    "aggregate_lists_per_turn",
    "calculate_cogmap_per_turn",
    "compute_error_aggregates",
    "compute_correctness_aggregates",
    "compute_consistency_aggregates",
]


