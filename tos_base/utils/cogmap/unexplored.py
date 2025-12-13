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
    
    Overall score: F1 of precision/recall.
    
    Args:
        predicted_coords: List of (x, y) coordinates predicted as unexplored
        correct_coords: List of ground truth unexplored (x, y) coordinates
        
    Returns:
        UnexploredMetrics with precision, recall, and overall scores
    """
    assert correct_coords, "No correct coordinates provided"
    
    if not predicted_coords:
        # No predictions when there are unexplored regions
        return UnexploredMetrics(precision=0.0, recall=0.0, overall=0.0, valid=True)
    
    correct_set = set(correct_coords)
    predicted_set = set(predicted_coords)
    
    # Calculate correct and wrong predictions
    correct_predictions = len(predicted_set & correct_set)
    wrong_predictions = len(predicted_set - correct_set)
    total_correct = len(correct_set)
    
    # Calculate precision and recall
    precision = correct_predictions / len(predicted_set) if predicted_set else 0.0
    recall = correct_predictions / total_correct if total_correct else 0.0
    
    # Overall score: F1
    overall = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return UnexploredMetrics(
        precision=precision,
        recall=recall,
        overall=overall,
        valid=True,
    )


def parse_unexplored_response(text: str) -> List[Tuple[int, int]]:
    """Parse predicted coordinates from an LLM response.

    The input is expected to follow the unexplored prompt output format, i.e. a JSON object:
    {"unexplored": "(5, 3); (2, 1); (10, 2)"}.
    This parser is intentionally more permissive to handle common formatting variations.

    Supported forms include:
    - JSON: {"unexplored": "(1,2); (3,4)"} or {"unexplored": [[1,2],[3,4]]}
    - Plain text: "(1,2), (3,4)" / "[1, 2]; [3, 4]" / mixed separators

    Returns: list[(x, y)] (possibly empty).
    """
    import json
    import re

    def _extract_json_candidates(s: str) -> List[str]:
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        candidates = list(fenced) if fenced else []
        if candidates:
            return candidates
        # Fallback: scan for outermost balanced braces
        stack, start = [], None
        for i, ch in enumerate(s):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(s[start:i + 1])
                    start = None
        return candidates

    def _parse_pairs_from_string(s: str) -> List[Tuple[int, int]]:
        # (1,2) or [1,2] or {1,2} with optional spaces; allow negative ints
        pair_pat = r"[\(\[\{]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]\}]"
        out: List[Tuple[int, int]] = []
        for x_str, y_str in re.findall(pair_pat, s):
            try:
                out.append((int(x_str), int(y_str)))
            except Exception:
                continue
        # Also allow bare "1,2" pairs if nothing else matched
        if not out:
            bare_pat = r"(-?\d+)\s*,\s*(-?\d+)"
            for x_str, y_str in re.findall(bare_pat, s):
                try:
                    out.append((int(x_str), int(y_str)))
                except Exception:
                    continue
        # De-dup while keeping order
        seen = set()
        uniq: List[Tuple[int, int]] = []
        for p in out:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        return uniq

    def _parse_value(v) -> List[Tuple[int, int]]:
        if v is None:
            return []
        if isinstance(v, str):
            return _parse_pairs_from_string(v)
        if isinstance(v, (list, tuple)):
            coords: List[Tuple[int, int]] = []
            for it in v:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    try:
                        coords.append((int(it[0]), int(it[1])))
                    except Exception:
                        continue
                elif isinstance(it, str):
                    coords.extend(_parse_pairs_from_string(it))
            # De-dup while keeping order
            seen = set()
            uniq: List[Tuple[int, int]] = []
            for p in coords:
                if p in seen:
                    continue
                seen.add(p)
                uniq.append(p)
            return uniq
        if isinstance(v, dict):
            # Sometimes nested, e.g. {"coords": "..."}; best-effort: search within JSON text
            return _parse_pairs_from_string(json.dumps(v, ensure_ascii=False))
        return []

    if not isinstance(text, str):
        return []
    raw = text.strip()

    # Try JSON first
    json_dict = None
    for cand in _extract_json_candidates(raw):
        try:
            json_dict = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    if isinstance(json_dict, dict):
        return _parse_value(json_dict.get("unexplored"))

    # Fallback: plain text parsing
    return _parse_value(raw)


__all__ = [
    'compute_unexplored_regions',
    'generate_labeled_points',
    'evaluate_unexplored_predictions',
    'parse_unexplored_response',
    'UnexploredMetrics',
]
