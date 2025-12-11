"""
Unexplored Areas Evaluation Module

Evaluates predictions of unexplored grid coordinates against ground truth unexplored regions.
Uses connectivity analysis to determine if predictions correctly identify distinct unexplored areas.
"""

from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np
import random
from collections import deque

from .types import UnexploredMetrics


def generate_labeled_points(
    unexplored_regions: List[Set[Tuple[int, int]]],
    explored_positions: Set[Tuple[int, int]],
    num_distractors: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, Tuple[int, int], bool]], List[int]]:
    """Generate labeled points: one representative per unexplored region + distractor points.
    
    Args:
        unexplored_regions: List of connected unexplored region sets
        explored_positions: Set of already explored positions (for selecting distractors)
        num_distractors: Number of distractor points (defaults to same as number of regions)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - List of (label, (x, y), is_unexplored) tuples, randomly shuffled
        - List of correct labels (unexplored region labels)
    """
    if seed is not None:
        random.seed(seed)
    
    if num_distractors is None:
        num_distractors = len(unexplored_regions)
    
    labeled_points: List[Tuple[int, Tuple[int, int], bool]] = []
    correct_labels: List[int] = []
    
    # Select one point from each unexplored region
    for i, region in enumerate(unexplored_regions):
        label = i + 1  # Labels start from 1
        point = random.choice(list(region))
        labeled_points.append((label, point, True))
        correct_labels.append(label)
    
    # Select distractor points from explored positions
    if explored_positions and num_distractors > 0:
        available_distractors = list(explored_positions)
        if len(available_distractors) > num_distractors:
            distractor_points = random.sample(available_distractors, num_distractors)
        else:
            distractor_points = available_distractors
        
        start_label = len(unexplored_regions) + 1
        for i, point in enumerate(distractor_points):
            label = start_label + i
            labeled_points.append((label, point, False))
    
    # Shuffle to randomize order
    random.shuffle(labeled_points)
    
    return labeled_points, correct_labels


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
    predicted_coords: List[Tuple[int, int]],
    correct_coords: List[Tuple[int, int]],
) -> UnexploredMetrics:
    """Evaluate predicted unexplored coordinates against ground truth.
    
    Scoring formula: (correct answers / total correct) - (wrong answers / total correct)
    
    Args:
        predicted_coords: List of (x, y) coordinates predicted as unexplored
        correct_coords: List of ground truth unexplored (x, y) coordinates
        
    Returns:
        UnexploredMetrics with precision, recall, and overall scores
    """
    if not correct_coords:
        # No unexplored regions exist
        if not predicted_coords:
            # Correctly predicted empty
            return UnexploredMetrics(precision=1.0, recall=1.0, region_diversity=1.0, overall=1.0, valid=True)
        else:
            # Predicted coords when there are none
            return UnexploredMetrics(precision=0.0, recall=1.0, region_diversity=0.0, overall=0.0, valid=True)
    
    if not predicted_coords:
        # No predictions when there are unexplored regions
        return UnexploredMetrics(precision=0.0, recall=0.0, region_diversity=0.0, overall=0.0, valid=True)
    
    correct_set = set(correct_coords)
    predicted_set = set(predicted_coords)
    
    # Calculate correct and wrong predictions
    correct_predictions = len(predicted_set & correct_set)
    wrong_predictions = len(predicted_set - correct_set)
    total_correct = len(correct_set)
    
    # Calculate precision and recall
    precision = correct_predictions / len(predicted_set) if predicted_set else 0.0
    recall = correct_predictions / total_correct if total_correct else 0.0
    
    # Overall score: (correct / total_correct) - (wrong / total_correct)
    overall = (correct_predictions / total_correct) - (wrong_predictions / total_correct)
    
    # Region diversity (always 1.0 for coordinate format)
    region_diversity = 1.0
    
    return UnexploredMetrics(
        precision=precision,
        recall=recall,
        region_diversity=region_diversity,
        overall=overall,
        valid=True,
    )


def parse_unexplored_response(text: str) -> Optional[List[Tuple[int, int]]]:
    """Parse unexplored coordinates from LLM JSON response.
    
    Expected JSON format:
    {
        "unexplored": "(5, 3); (2, 1); (10, 2)"
    }
    
    Or:
    {
        "unexplored": "none"
    }
    
    Or:
    {
        "unexplored": "unknown"
    }
    
    Args:
        text: JSON response from LLM
        
    Returns:
        List of (x, y) coordinate tuples, or None if "unknown", or empty list if "none"
    """
    import re
    import json
    
    if not isinstance(text, str):
        return []
    
    text = text.strip()
    
    # Try to extract JSON from the text
    json_dict = None
    
    # Try fenced blocks first
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced if fenced else []
    
    # Fallback: scan for outermost balanced braces
    if not candidates:
        stack, start = [], None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i+1])
                    start = None
    
    # Try to parse JSON
    for cand in candidates:
        try:
            json_dict = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    
    if not json_dict or not isinstance(json_dict, dict):
        # Fallback: try to parse as plain text
        text_lower = text.lower()
        if "unknown" in text_lower:
            return None
        if "none" in text_lower:
            return []
        return []
    
    # Extract the "unexplored" field
    unexplored_value = json_dict.get("unexplored", "")
    
    if not isinstance(unexplored_value, str):
        return []
    
    unexplored_str = unexplored_value.strip().lower()
    
    # Handle special values
    if unexplored_str == "unknown":
        return None
    
    if unexplored_str == "none" or not unexplored_str:
        return []
    
    # Parse coordinates in format (x, y); (x, y); ...
    coords: List[Tuple[int, int]] = []
    pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
    matches = re.findall(pattern, unexplored_value)
    
    for match in matches:
        try:
            x, y = int(match[0]), int(match[1])
            coords.append((x, y))
        except ValueError:
            continue
    
    return coords


__all__ = [
    'compute_unexplored_regions',
    'generate_labeled_points',
    'evaluate_unexplored_predictions',
    'parse_unexplored_response',
    'UnexploredMetrics',
]
