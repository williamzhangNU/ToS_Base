import base64
from io import BytesIO
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fig_to_data_uri(fig) -> str:
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def create_infogain_plot(infogain_per_turn: List[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    turns = list(range(1, len(infogain_per_turn) + 1))
    ax.plot(turns, infogain_per_turn, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Average Information Gain')
    ax.set_title(f'Average Information Gain per Turn - {title}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(infogain_per_turn) + 0.5)
    return _fig_to_data_uri(fig)


def create_cogmap_metrics_plot(
    series: Dict[str, List[Optional[float]]],
    title: str,
    include_dir: bool = True,
    include_facing: bool = True,
    include_pos: bool = True,
    include_overall: bool = True,
) -> Optional[str]:
    keys = [
        ('dir', include_dir, 'Direction'),
        ('facing', include_facing, 'Facing'),
        ('pos', include_pos, 'Position'),
        ('overall', include_overall, 'Overall'),
    ]

    any_data = any(
        include and isinstance(series.get(k), list) and any(v is not None for v in series.get(k, []))
        for k, include, _ in keys
    )
    if not any_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    turns = None
    for k, include, label in keys:
        if not include:
            continue
        vals = series.get(k, [])
        if not isinstance(vals, list) or len(vals) == 0:
            continue
        y = [np.nan if v is None else float(v) for v in vals]
        if turns is None:
            turns = list(range(1, len(y) + 1))
        ax.plot(turns, y, marker='o', linewidth=2, markersize=3, label=label)

    if turns is None:
        plt.close(fig)
        return None

    ax.set_xlabel('Turn')
    ax.set_ylabel('Similarity')
    ax.set_title(f'Cognitive Map Similarity per Turn - {title}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(turns) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return _fig_to_data_uri(fig)


def visualize_json(json_path: str, output_html: str, show_images: bool = True) -> str:
    # Local import to avoid circular dependency
    from .visualization import Visualization
    viz = Visualization(json_path, output_html, show_images)
    return viz.visualize()


