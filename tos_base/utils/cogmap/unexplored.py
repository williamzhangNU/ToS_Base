"""
Unexplored Areas Evaluation Module

Evaluates predictions of unexplored grid coordinates against ground truth unexplored regions.
Uses connectivity analysis to determine if predictions correctly identify distinct unexplored areas.
"""

from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np
from collections import deque

from .types import UnexploredMetrics


def compute_unexplored_regions(
    observed_positions: Set[Tuple[int, int]],
    room_bounds: Tuple[int, int, int, int],
) -> List[Set[Tuple[int, int]]]:
    """Compute connected unexplored regions in the grid.
    
    Args:
        observed_positions: Set of (x, y) coordinates that have been observed
        room_bounds: (min_x, max_x, min_y, max_y) to define the search area
        
    Returns:
        List of sets, each set containing (x, y) coordinates of a connected unexplored region
    """
    min_x, max_x, min_y, max_y = room_bounds
    
    # Build set of all grid cells
    all_cells = {(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)}
    
    # Unexplored cells = all cells - observed positions
    unexplored = all_cells - observed_positions
    
    if not unexplored:
        return []
    
    # Find connected components using BFS
    regions: List[Set[Tuple[int, int]]] = []
    visited: Set[Tuple[int, int]] = set()
    
    def bfs(start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """BFS to find all connected unexplored cells from start."""
        region = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            region.add((x, y))
            
            # Check 4-connected neighbors (can change to 8-connected if needed)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in unexplored and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return region
    
    for cell in unexplored:
        if cell not in visited:
            region = bfs(cell)
            if region:
                regions.append(region)
    
    return regions


def point_in_region(point: Tuple[int, int], region: Set[Tuple[int, int]]) -> bool:
    """Check if a point is within a region."""
    return point in region


def evaluate_unexplored_predictions(
    predicted_points: List[Tuple[int, int]],
    unexplored_regions: List[Set[Tuple[int, int]]],
) -> UnexploredMetrics:
    """Evaluate predicted unexplored points against ground truth regions.
    
    Args:
        predicted_points: List of (x, y) coordinates predicted as unexplored
        unexplored_regions: List of connected unexplored region sets
        
    Returns:
        UnexploredMetrics with precision, recall, region_diversity, and overall scores
    """
    if not unexplored_regions:
        # No unexplored regions exist
        if not predicted_points:
            # Correctly predicted empty
            return UnexploredMetrics(precision=1.0, recall=1.0, region_diversity=1.0, overall=1.0, valid=True)
        else:
            # Predicted points when there are none
            return UnexploredMetrics(precision=0.0, recall=1.0, region_diversity=0.0, overall=0.0, valid=True)
    
    if not predicted_points:
        # No predictions when there are unexplored regions
        return UnexploredMetrics(precision=0.0, recall=0.0, region_diversity=0.0, overall=0.0, valid=True)
    
    # Calculate precision: fraction of predicted points in unexplored regions
    points_in_unexplored = 0
    covered_region_indices: Set[int] = set()
    point_to_region: Dict[Tuple[int, int], int] = {}  # Track which region each point belongs to
    
    for point in predicted_points:
        for i, region in enumerate(unexplored_regions):
            if point_in_region(point, region):
                points_in_unexplored += 1
                covered_region_indices.add(i)
                point_to_region[point] = i
                break  # A point can only be in one region
    
    precision = points_in_unexplored / len(predicted_points)
    
    # Calculate recall: fraction of unexplored regions that have at least one prediction
    recall = len(covered_region_indices) / len(unexplored_regions)
    
    # Calculate region diversity: are predictions in different regions?
    # If all valid predictions are in the same region, diversity is low
    if points_in_unexplored > 0:
        unique_regions_hit = len(set(point_to_region.values()))
        max_possible = min(points_in_unexplored, len(unexplored_regions))
        region_diversity = unique_regions_hit / max_possible if max_possible > 0 else 0.0
    else:
        region_diversity = 0.0
    
    # Overall: harmonic mean of precision and recall (F1-like)
    if precision + recall > 0:
        overall = 2 * precision * recall / (precision + recall)
    else:
        overall = 0.0
    
    return UnexploredMetrics(
        precision=precision,
        recall=recall,
        region_diversity=region_diversity,
        overall=overall,
        valid=True,
    )


def parse_unexplored_response(json_data: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Parse unexplored points from LLM response JSON.
    
    Expected format:
    {
        "unexplored_points": [[x1, y1], [x2, y2], ...]
    }
    
    Args:
        json_data: Parsed JSON from LLM response
        
    Returns:
        List of (x, y) coordinate tuples
    """
    points: List[Tuple[int, int]] = []
    
    if not isinstance(json_data, dict):
        return points
    
    raw_points = json_data.get('unexplored_points', [])
    if not isinstance(raw_points, list):
        return points
    
    for pt in raw_points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                x, y = int(pt[0]), int(pt[1])
                points.append((x, y))
            except (ValueError, TypeError):
                continue
    
    return points


def compute_observed_positions_from_solver(
    possible_positions: Dict[str, List[List[int]]],
    grid_size: int,
) -> Set[Tuple[int, int]]:
    """Compute observed positions based on spatial solver constraints.
    
    A position is considered "observed" if it's constrained by object observations.
    We use the possible positions from the solver to infer observed areas.
    
    Args:
        possible_positions: Dict mapping object names to lists of [x, y] possible positions
        grid_size: Size of the grid
        
    Returns:
        Set of (x, y) coordinates that are considered "observed"
    """
    observed: Set[Tuple[int, int]] = set()
    
    for name, positions in possible_positions.items():
        if not positions:
            continue
        for pos in positions:
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                try:
                    observed.add((int(pos[0]), int(pos[1])))
                except (ValueError, TypeError):
                    continue
    
    return observed


def compute_observed_positions_from_visibility(
    agent_positions_history: List[Tuple[int, int]],
    agent_orientations_history: List[Tuple[int, int]],
    grid_size: int,
    fov_angle: int = 90,
    view_distance: int = 5,
) -> Set[Tuple[int, int]]:
    """Compute observed positions based on agent's visibility history.
    
    For each position/orientation in history, compute which grid cells were visible.
    
    Args:
        agent_positions_history: List of agent (x, y) positions over time
        agent_orientations_history: List of agent orientation vectors
        grid_size: Size of the grid
        fov_angle: Field of view angle in degrees
        view_distance: Maximum view distance
        
    Returns:
        Set of (x, y) coordinates that have been observed
    """
    observed: Set[Tuple[int, int]] = set()
    
    half_fov = np.radians(fov_angle / 2)
    
    for pos, ori in zip(agent_positions_history, agent_orientations_history):
        px, py = pos
        ox, oy = ori
        
        # Normalize orientation
        ori_len = np.sqrt(ox**2 + oy**2)
        if ori_len == 0:
            continue
        ox, oy = ox / ori_len, oy / ori_len
        
        # Check all cells within view distance
        for dx in range(-view_distance, view_distance + 1):
            for dy in range(-view_distance, view_distance + 1):
                tx, ty = px + dx, py + dy
                
                # Check distance
                dist = np.sqrt(dx**2 + dy**2)
                if dist > view_distance or dist == 0:
                    continue
                
                # Check angle
                to_target = np.array([dx / dist, dy / dist])
                dot = ox * to_target[0] + oy * to_target[1]
                
                # Within FOV
                if dot >= np.cos(half_fov):
                    observed.add((int(tx), int(ty)))
        
        # Agent's current position is always observed
        observed.add((int(px), int(py)))
    
    return observed


__all__ = [
    'compute_unexplored_regions',
    'evaluate_unexplored_predictions',
    'parse_unexplored_response',
    'compute_observed_positions_from_solver',
    'compute_observed_positions_from_visibility',
    'UnexploredMetrics',
]
