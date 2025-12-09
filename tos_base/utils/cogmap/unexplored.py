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
    unexplored_positions: Set[Tuple[int, int]],
) -> List[Set[Tuple[int, int]]]:
    """Compute connected unexplored regions in the grid using BFS.
    
    Args:
        unexplored_positions: Set of (x, y) coordinates that are unexplored
        
    Returns:
        List of sets, each set containing (x, y) coordinates of a connected unexplored region
    """
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
                if (nx, ny) in unexplored_positions and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return region
    
    for cell in unexplored_positions:
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


def parse_unexplored_response(json_data: Dict[str, Any]) -> Dict[str, Optional[List[Tuple[int, int]]]]:
    """Parse unexplored points from LLM response JSON (multi-room format).
    
    Expected format:
    {
        "1": [[x1, y1], [x2, y2], ...],
        "2": "unknown"
    }
    
    Args:
        json_data: Parsed JSON from LLM response
        
    Returns:
        Dict mapping room_id (str) to:
        - list of (x, y) coordinate tuples if room was observed
        - None if room value is "unknown" (not visited)
    """
    result: Dict[str, Optional[List[Tuple[int, int]]]] = {}
    
    if not isinstance(json_data, dict):
        return result
    
    # Multi-room format: keys are room IDs
    for room_id_key, raw_value in json_data.items():
        # Skip non-room-id keys (room IDs should be numeric strings)
        try:
            int(room_id_key)
        except ValueError:
            continue
        
        # Handle "unknown" value
        if isinstance(raw_value, str) and raw_value.lower() == "unknown":
            result[room_id_key] = None
            continue
        
        # Parse coordinate list
        points: List[Tuple[int, int]] = []
        if isinstance(raw_value, list):
            for pt in raw_value:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        x, y = int(pt[0]), int(pt[1])
                        points.append((x, y))
                    except (ValueError, TypeError):
                        continue
        result[room_id_key] = points
    
    return result


__all__ = [
    'compute_unexplored_regions',
    'evaluate_unexplored_predictions',
    'parse_unexplored_response',
    'UnexploredMetrics',
]
