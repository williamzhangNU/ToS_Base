from typing import Dict, List, Any, Tuple, Optional
import numpy as np


def calculate_other_candidates_metrics(env_data: Dict[str, Any], max_candidates: int = 3) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Per-turn metrics for `other_candidates` vs GT `possible_positions`.

    Returns (other_candidates_f1_per_turn, other_candidates_p_per_turn, other_candidates_r_per_turn, other_candidates_count_per_turn).
    """
    turn_logs = env_data.get("env_turn_logs") or []
    f1_per_turn: List[float] = []
    p_per_turn: List[float] = []
    r_per_turn: List[float] = []
    count_score_per_turn: List[float] = []

    def _to_coord_set(v: Any) -> set[tuple[int, int]]:
        if not v:
            return set()
        out: set[tuple[int, int]] = set()
        if isinstance(v, (list, tuple)):
            for it in v:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    try:
                        out.add((int(it[0]), int(it[1])))
                    except Exception:
                        continue
        return out

    def _to_coord(v: Any) -> Optional[tuple[int, int]]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                return (int(v[0]), int(v[1]))
            except Exception:
                return None
        return None

    def _score(pred: set[tuple[int, int]], gt_other: set[tuple[int, int]]) -> tuple[float, float, float, float]:
        """Return (f1, precision, recall, count_score) in [0,1]."""
        expected = min(int(max_candidates), len(gt_other))
        if expected == 0:
            return (1.0, 1.0, 1.0, 1.0) if len(pred) == 0 else (0.0, 0.0, 0.0, 0.0)
        inter = len(pred & gt_other)
        precision = (inter / len(pred)) if pred else 0.0
        recall = (min(inter, expected) / expected) if expected > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        count_score = (min(len(pred), expected) / max(len(pred), expected)) if max(len(pred), expected) > 0 else 0.0
        return float(f1 * count_score), float(precision), float(recall), float(count_score)

    for log in turn_logs:
        if not log.get("is_exploration_phase") or not log.get("cogmap_log"):
            continue
        cog_log = log.get("cogmap_log") or {}
        global_log = cog_log.get("global") or {}
        if not global_log.get("extraction_success"):
            f1_per_turn.append(0.0)
            p_per_turn.append(0.0)
            r_per_turn.append(0.0)
            count_score_per_turn.append(0.0)
            continue

        pred_json = global_log.get("pred_json") or {}
        possible_positions = log['exploration_log']['possible_positions']

        pred_by_name: Dict[str, Any] = {}
        if isinstance(pred_json, dict):
            for k, v in pred_json.items():
                if isinstance(k, str):
                    pred_by_name[k.replace("_", " ").strip()] = v

        scores: List[float] = []
        precs: List[float] = []
        recs: List[float] = []
        count_scores: List[float] = []
        if isinstance(possible_positions, dict):
            for gt_name, gt_pts in possible_positions.items():
                gt_all = _to_coord_set(gt_pts)
                obj_data = pred_by_name.get(str(gt_name).replace("_", " ").strip())

                pred_pos = _to_coord((obj_data or {}).get("position") if isinstance(obj_data, dict) else None)
                gt_other = set(gt_all)
                if pred_pos in gt_other:
                    gt_other.remove(pred_pos)  # enforce "other" (exclude `position`)

                pred_other = _to_coord_set((obj_data or {}).get("other_candidates") if isinstance(obj_data, dict) else None)
                f1, p, r, cs = _score(pred_other, gt_other)
                scores.append(f1)
                precs.append(p)
                recs.append(r)
                count_scores.append(cs)

        f1_per_turn.append(float(np.mean(scores)) if scores else 0.0)
        p_per_turn.append(float(np.mean(precs)) if precs else 0.0)
        r_per_turn.append(float(np.mean(recs)) if recs else 0.0)
        count_score_per_turn.append(float(np.mean(count_scores)) if count_scores else 0.0)

    return f1_per_turn, p_per_turn, r_per_turn, count_score_per_turn


