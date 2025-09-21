from typing import Dict, Any, List
import numpy as np
from scipy.stats import pearsonr


def compute_correlation_metrics(env_data_list: Dict, exp_type: str = 'active') -> Dict[str, Any]:
    """
    Compute correlations between cognitive map metrics and evaluation metrics, information gain metrics.

    Args:
        env_data_list: Environment data list
        exp_type: Task type, 'active' or 'passive'

    Returns:
        Dictionary containing all correlation analysis results
    """
    assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"
    if exp_type == 'passive':
        return {}

    last_global_vs_gt_fulls = []
    evaluation_metric_list = {}
    last_infogains = []

    for s in env_data_list:
        metrics = s.get('metrics')
        cogmap_metric = metrics.get('cogmap')
        evaluation_metric = metrics.get('evaluation')
        exploration_metric = metrics.get('exploration', {})
        assert isinstance(cogmap_metric, dict) and isinstance(evaluation_metric, dict), "Each env_data must have 'cogmap' and 'evaluation' metrics"

        # Extract last_global_vs_gt_full metric
        exploration = cogmap_metric.get('exploration', {})
        correctness = exploration.get('correctness', {})
        last_global_full = correctness.get('last_global_vs_gt_full', {})
        overall_cogmap = last_global_full.get('overall', 0.0)

        # Extract last_infogain metric
        last_infogain = exploration_metric.get('final_information_gain', 0.0)

        if isinstance(overall_cogmap, (int, float)) and not np.isnan(overall_cogmap):
            last_global_vs_gt_fulls.append(float(overall_cogmap))

            # Extract evaluation metrics
            # Overall accuracy
            avg_accuracy = evaluation_metric.get('overall', {}).get('avg_accuracy')
            if 'avg_accuracy' not in evaluation_metric_list:
                evaluation_metric_list['avg_accuracy'] = []
            evaluation_metric_list['avg_accuracy'].append(float(avg_accuracy))

            # Accuracy for each task
            per_task = evaluation_metric.get('per_task', {})
            for task_name, task_metrics in per_task.items():
                task_acc = task_metrics.get('accuracy', 0.0)
                if task_name not in evaluation_metric_list:
                    evaluation_metric_list[task_name] = []
                evaluation_metric_list[task_name].append(float(task_acc))

            # Add information gain data
            if isinstance(last_infogain, (int, float)) and not np.isnan(last_infogain):
                last_infogains.append(float(last_infogain))
            else:
                last_infogains.append(0.0)  # Use default value to maintain consistent length

    cogmap_acc_correlations = {}
    for task_name, evaluation_values in evaluation_metric_list.items():
        cogmap_acc_correlations[task_name] = calculate_pearson_correlation(last_global_vs_gt_fulls, evaluation_values)

    cogmap_infogain_correlation = calculate_pearson_correlation(last_global_vs_gt_fulls, last_infogains)

    return {
        'cogmap_acc_correlations': cogmap_acc_correlations,
        'cogmap_infogain_correlation': cogmap_infogain_correlation,
        'last_global_vs_gt_fulls': last_global_vs_gt_fulls,
        'last_infogains': last_infogains,
        'avg_acc_metrics': evaluation_metric_list.get('avg_accuracy', []),
        'n_samples': len(last_global_vs_gt_fulls)
    }


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Dict[str, Any]:
    assert len(x) == len(y), "Length of x and y must be the same"
    try:
        corr_coef, p_value = pearsonr(x, y)
        return {
            'pearson_r': float(corr_coef),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'n_samples': len(x)
        }
    except Exception as e:
        return {
            'pearson_r': None,
            'p_value': None,
            'significant': False,
            'n_samples': len(x),
            'error': str(e)
        }

