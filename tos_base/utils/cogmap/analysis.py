from typing import List, Dict, Any, Callable, Optional
from .types import MapCogMetrics, RelationMetrics

from .types import MapCogMetrics, RelationMetrics


def _avg(values: List[float]) -> float:
    v = [x for x in values if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else 0.0


def _avg_metrics(keys: List[str], metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {k: 0.0 for k in keys}
    return {k: _avg([float(m.get(k, 0.0)) for m in metrics_list if isinstance(m, dict)]) for k in keys}


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


# ---- Dataclass-based aggregation helpers ----
def _avg_map_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    mats: List[MapCogMetrics] = []
    for d in dicts:
        m = MapCogMetrics.from_dict(d)
        if m.valid:
            mats.append(m)
    return MapCogMetrics.average(mats).to_dict() if mats else MapCogMetrics.invalid().to_dict()


def _avg_rel_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    rms: List[RelationMetrics] = []
    for d in dicts:
        r = RelationMetrics.from_dict(d)
        if r.valid:
            rms.append(r)
    return RelationMetrics.average(rms).to_dict() if rms else RelationMetrics.invalid().to_dict()


def _avg_map_over_turns(env_data: Dict[str, Any], section: str, field: str) -> MapCogMetrics:
    mats: List[MapCogMetrics] = []
    for t in env_data.get('env_turn_logs', []):
        d = (t.get('cogmap_log') or {}).get(section, {}).get(field, {})
        m = MapCogMetrics.from_dict(d)
        if m.valid:
            mats.append(m)
    return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()


def compute_error_aggregates(env_data_list: List[Dict[str, Any]],) -> Dict[str, Any]:

    def _per_sample_local_err(env_data: Dict[str, Any]) -> Dict[str, float]:
        m = _avg_map_over_turns(env_data, section='local', field='metrics')
        return m.to_dict()

    def _per_sample_global_err(env_data: Dict[str, Any]) -> Dict[str, float]:
        m = _avg_map_over_turns(env_data, section='global', field='metrics')
        return m.to_dict()

    return {
        'local_vs_gt_local_avg': aggregate_per_sample_then_group(env_data_list, _per_sample_local_err),
        'global_vs_gt_global_avg': aggregate_per_sample_then_group(env_data_list, _per_sample_global_err),
    }


def get_last_exploration_cogmap(env_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the last exploration turn's cogmap_log if present."""
    for t in reversed(env_data.get('env_turn_logs', [])):
        if t.get('cogmap_log'):
            return t.get('cogmap_log')
    return None


def compute_correctness_aggregates(env_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:


    # Passive global map (full) aggregated
    passive_vals = []
    for env_data in env_data_list:
        cfg = (env_data.get('env_info') or {}).get('config') or {}
        if cfg.get('exp_type') == 'passive':
            turns = env_data.get('env_turn_logs') or []
            if turns:
                mm = (turns[0].get('cogmap_log') or {}).get('global', {}).get('metrics_full', {})
                if mm:
                    passive_vals.append(MapCogMetrics.from_dict(mm).to_dict())

    # Dataclass-based sample averaging
    last_global_vals = []
    last_rel_vals = []
    for env_data in env_data_list:
        lg = get_last_exploration_cogmap(env_data)
        if lg:
            last_global_vals.append(MapCogMetrics.from_dict((lg or {}).get('global', {}).get('metrics_full', {})).to_dict())
            last_rel_vals.append(RelationMetrics.from_dict((lg or {}).get('relations', {}).get('metrics_full', {})).to_dict())

    return {
        'last_global_vs_gt_full': _avg_map_dicts(last_global_vals),
        'last_relations_vs_gt_full': _avg_rel_dicts(last_rel_vals),
        'per_turn_global_full': calculate_cogmap_per_turn(env_data_list, mode='full'),
        'per_turn_global_observed': calculate_cogmap_per_turn(env_data_list, mode='update'),
        'passive_global_full': _avg_map_dicts(passive_vals),
    }


def compute_consistency_aggregates(env_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:

    def _per_sample_local_vs_global(env_data: Dict[str, Any]) -> Dict[str, float]:
        mats: List[MapCogMetrics] = []
        for t in env_data.get('env_turn_logs', []):
            cons = (t.get('cogmap_log') or {}).get('consistency') or {}
            m = MapCogMetrics.from_dict(cons.get('local_vs_global') or {})
            if m.valid:
                mats.append(m)
        return MapCogMetrics.average(mats).to_dict() if mats else MapCogMetrics.invalid().to_dict()

    def _last_rooms_vs_global(env_data: Dict[str, Any]) -> Dict[str, float]:
        lg = get_last_exploration_cogmap(env_data)
        m = MapCogMetrics.from_dict(((lg or {}).get('consistency', {}).get('rooms_vs_global', {}).get('average', {})))
        return m.to_dict()

    def _last_map_vs_rel(env_data: Dict[str, Any]) -> float:
        lg = get_last_exploration_cogmap(env_data)
        return float((lg or {}).get('consistency', {}).get('map_vs_relations', 0.0))

    def _last_rel_cons(env_data: Dict[str, Any]) -> float:
        lg = get_last_exploration_cogmap(env_data)
        return float((lg or {}).get('consistency', {}).get('relations_consistency', 0.0))

    # Dataclass-based aggregation across samples
    local_vs_global_avg = aggregate_per_sample_then_group(env_data_list, _per_sample_local_vs_global)
    rooms_vals = [_last_rooms_vs_global(s) for s in env_data_list]
    rooms_vs_global_last = _avg_map_dicts(rooms_vals)
    map_vs_relations_last = (sum([_last_map_vs_rel(s) for s in env_data_list]) / len(env_data_list)) if env_data_list else 0.0
    relations_consistency_last = (sum([_last_rel_cons(s) for s in env_data_list]) / len(env_data_list)) if env_data_list else 0.0

    return {
        'local_vs_global_avg': local_vs_global_avg,
        'rooms_vs_global_last': rooms_vs_global_last,
        'map_vs_relations_last': map_vs_relations_last,
        'relations_consistency_last': relations_consistency_last,
    }


def compute_evaluation_correctness_aggregates(env_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluation (non-exploration) global correctness across samples."""
    vals: List[Dict[str, float]] = []
    for env_data in env_data_list:
        eval_tasks = env_data.get('evaluation_tasks') or {}
        for task_log in eval_tasks.values():
            m = (task_log.get('cogmap_log') or {}).get('global', {}).get('metrics_full', {})
            if m:
                vals.append(MapCogMetrics.from_dict(m).to_dict())
    return _avg_map_dicts(vals)


__all__ = [
    "aggregate_per_sample_then_group",
    "aggregate_lists_per_turn",
    "calculate_cogmap_per_turn",
    "compute_error_aggregates",
    "compute_correctness_aggregates",
    "compute_consistency_aggregates",
]


