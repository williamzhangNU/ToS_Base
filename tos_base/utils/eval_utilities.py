"""Evaluation helpers for spatial tasks."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from types import SimpleNamespace
import math
import re
import numpy as np

from ..core.relationship import PairwiseRelationshipDiscrete

if TYPE_CHECKING:
    from .cogmap.metrics import compute_pos_sim


# ========== Constants ==========
_COORD_TOL = 1e-3
_BACKWARD_LOC_DEFAULT_THRESHOLD = 0.8
_E2A_DEFAULT_SIM_THRESHOLD = 0.9
_COS_45 = float(np.cos(np.deg2rad(45)))
_ORI_TO_DEG = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
_DEG_TO_ORI = {deg: ori for ori, deg in _ORI_TO_DEG.items()}

# Label aliases for direction and distance
_LABEL_ALIASES: Dict[str, Sequence[str]] = {
    "front-left": ("front-left", "left-front", "front left", "left front", "frontleft", "leftfront"),
    "front-right": ("front-right", "right-front", "front right", "right front", "frontright", "rightfront"),
    "front-slight-left": ("front-slight-left", "front slightly left", "slightly front left", "frontslightleft", "slight front left"),
    "front-slight-right": ("front-slight-right", "front slightly right", "slightly front right", "frontslightright", "slight front right"),
    "mid distance": ("mid distance", "mid-distance", "mid", "medium", "medium distance"),
    "same distance": ("same distance", "same", "same-distance"),
    "slightly far": ("slightly far", "slightly-far"),
    "very far": ("very far", "very-far"),
    "extremely far": ("extremely far", "extremely-far", "extreme", "extreme far"),
}


# ========== Text Normalization ==========
def _normalize_joined(text: str) -> str:
    """Remove whitespace and hyphens, convert to lowercase."""
    return re.sub(r"[\s-]+", "", text.lower())

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace and convert to lowercase."""
    return re.sub(r"\s+", " ", text.strip()).lower()

def _casefold(value: Any) -> str:
    """Convert any value to lowercase string."""
    return str(value).strip().lower()

def _require_text(value: Any) -> Optional[str]:
    """Return stripped text if non-empty, else None."""
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None


# ========== Label Matching System ==========
_LABEL_LOOKUP: Dict[str, str] = {}
_LABEL_COMPONENTS: Dict[str, set[str]] = {}

def _build_label_system() -> None:
    """Build label alias lookup and component tables."""
    for canonical, variants in _LABEL_ALIASES.items():
        canonical_norm = _normalize_joined(canonical)
        _LABEL_LOOKUP[canonical_norm] = canonical_norm
        
        parts = {_normalize_joined(token) for token in re.split(r"[-\s]+", canonical) if token}
        parts.add(canonical_norm)
        _LABEL_COMPONENTS[canonical_norm] = parts
        
        for variant in variants:
            _LABEL_LOOKUP[_normalize_joined(variant)] = canonical_norm

_build_label_system()

def _canonicalize_label(label: Any) -> str:
    """Convert label to canonical form."""
    norm = _normalize_joined(str(label)) if label is not None else ""
    return _LABEL_LOOKUP.get(norm, norm)

def _label_components(label: Any) -> set[str]:
    """Get component parts of a label."""
    canonical = _canonicalize_label(label)
    return _LABEL_COMPONENTS.get(canonical, {canonical})

def _labels_match(pred_label: Any, target_label: Any) -> bool:
    """Check if two labels match (exact or partial component match)."""
    pred_canonical = _canonicalize_label(pred_label)
    target_canonical = _canonicalize_label(target_label)
    
    if not pred_canonical or not target_canonical:
        return False
    if pred_canonical == target_canonical:
        return True
    
    target_parts = _label_components(target_label)
    pred_parts = _label_components(pred_label)
    return (pred_canonical in target_parts) or (target_canonical in pred_parts)


# ========== Parsing Utilities ==========
def _extract_sequence(
    raw: Any,
    value_type: type = str,
    clean_pattern: str | None = None,
) -> Optional[List[Any]]:
    """Extract sequence from various formats (list, tuple, or string)."""
    if raw is None:
        return None
    
    # Direct list/tuple
    if isinstance(raw, (list, tuple)):
        try:
            return [value_type(item) for item in raw]
        except (TypeError, ValueError):
            return None
    
    # Parse from string
    text = str(raw)
    content = text
    
    # Extract from parentheses or brackets
    if '(' in text and ')' in text:
        match = re.search(r"\((.*?)\)", text)
        if match:
            content = match.group(1)
    elif '[' in text and ']' in text:
        match = re.search(r"\[(.*?)\]", text)
        if match:
            content = match.group(1)
    
    # Parse comma-separated values
    items: List[Any] = []
    for token in content.split(','):
        value = token.strip().strip("'\"")
        if not value:
            continue
        if clean_pattern:
            value = re.sub(clean_pattern, '', value, flags=re.IGNORECASE)
        try:
            if value_type is int:
                items.append(int(value))
            elif value_type is float:
                items.append(float(value))
            else:
                items.append(value_type(value) if value_type is not str else value)
        except (TypeError, ValueError):
            return None
    
    return items or None

def extract_elements(
    raw: Any,
    expected_type: type = str,
    clean_pattern: str | None = None,
) -> Optional[List[Any]]:
    """Public API for sequence extraction."""
    return _extract_sequence(raw, expected_type, clean_pattern)

def _parse_direction_distance(raw: str) -> Optional[Tuple[str, str]]:
    """Parse direction and distance from text."""
    # Try as comma-separated list
    items = _extract_sequence(raw, str)
    if items and len(items) >= 2:
        return items[0], items[1]
    
    # Try semicolon or newline separated
    segments = [seg.strip() for seg in re.split(r"[;\n]", str(raw)) if seg.strip()]
    if len(segments) >= 2:
        return segments[0], segments[1]
    
    # Try space/comma separated words
    words = [word.strip() for word in re.split(r"[ ,]", str(raw)) if word.strip()]
    if len(words) >= 2:
        return (' '.join(words[:-1]), words[-1]) if len(words) > 2 else (words[0], words[1])
    
    return None

def _parse_coordinate_list(raw: Any) -> Optional[List[Tuple[float, float]]]:
    """Parse list of (x, y) coordinates from text."""
    matches = re.findall(r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?", str(raw))
    if not matches:
        return None
    
    coords: List[Tuple[float, float]] = []
    for pair in matches:
        x_str, y_str = pair.split(',')
        try:
            coords.append((float(x_str.strip()), float(y_str.strip())))
        except ValueError:
            return None
    
    return coords or None

def _parse_indexed_lines(raw: Any) -> List[Tuple[int, str]]:
    """Parse numbered list format: '1. item'."""
    if not isinstance(raw, str):
        return []
    
    pairs: List[Tuple[int, str]] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or '. ' not in text:
            continue
        idx_part, value = text.split('. ', 1)
        try:
            idx = int(idx_part.strip()) - 1
        except ValueError:
            continue
        pairs.append((idx, value.strip()))
    
    return pairs

def _parse_coord_orientation(text: str) -> Optional[Tuple[Tuple[float, float], str]]:
    """Parse coordinate and orientation from text like '(x, y) facing north'."""
    if not isinstance(text, str):
        return None
    
    coord_match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", text)
    if not coord_match:
        return None
    
    coord = (float(coord_match.group(1)), float(coord_match.group(2)))
    
    ori_match = re.search(r'facing\s+([a-zA-Z-]+)', text, re.IGNORECASE)
    if ori_match:
        orientation = ori_match.group(1).strip().lower()
    else:
        fallback = re.findall(r'(north|east|south|west)', text, re.IGNORECASE)
        if not fallback:
            return None
        orientation = fallback[-1].strip().lower()
    
    return coord, orientation

def _parse_object_relations(text: str) -> List[Tuple[str, str, str]]:
    """Parse object relations in format: 'obj is at direction, distance'."""
    items: List[Tuple[str, str, str]] = []
    parts = [p.strip() for p in text.split(';') if p.strip()]
    
    for part in parts:
        match = re.match(r'(.+?)\s+is\s+at\s+(.+?),\s*(.+)', part.strip(), re.IGNORECASE)
        if match:
            obj_name = match.group(1).strip()
            direction = _canonicalize_label(match.group(2).strip())
            distance = _canonicalize_label(match.group(3).strip())
            items.append((obj_name.lower(), direction, distance))
    
    return items

def _parse_action_sequence(text: str) -> List[Tuple[str, Any]]:
    """Parse navigation actions: 'JumpTo(obj), Rotate(90)'."""
    actions: List[Tuple[str, Any]] = []
    
    for token in [p.strip() for p in str(text).split(',') if p.strip()]:
        jump = re.match(r'JumpTo\s*\(\s*(.+?)\s*\)', token, re.IGNORECASE)
        if jump:
            actions.append(('jumpto', jump.group(1).strip()))
            continue
        
        rotate = re.match(r'Rotate\s*\(\s*(-?\d+)\s*\)', token, re.IGNORECASE)
        if rotate:
            actions.append(('rotate', int(rotate.group(1))))
            continue
        
        return []  # Invalid format
    
    return actions


# ========== Geometry Utilities ==========
def _coerce_point(pt: Sequence[Any]) -> Tuple[float, float]:
    """Convert sequence to (x, y) float tuple."""
    if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) < 2:
        raise ValueError("invalid point")
    return float(pt[0]), float(pt[1])

def _points_close(a: Sequence[float], b: Sequence[float], tol: float = _COORD_TOL) -> bool:
    """Check if two points are within tolerance."""
    return math.dist(a, b) <= tol

def _coord_norm_from_gt(coords: Sequence[Sequence[float]]) -> float:
    """Calculate RMS norm from ground truth coordinates."""
    arr = np.array(coords, dtype=float)
    if arr.size == 0:
        return 1.0
    norms = (arr * arr).sum(axis=1)
    mean_norm = float(norms.mean()) if norms.size else 0.0
    return float(math.sqrt(mean_norm)) if mean_norm > 0 else 1.0

def _coords_to_room(coords: Sequence[Sequence[float]]) -> SimpleNamespace:
    """Convert coordinates to room-like object for metric computation."""
    objects = [
        SimpleNamespace(name=str(idx), pos=np.array(pt, dtype=float))
        for idx, pt in enumerate(coords)
    ]
    return SimpleNamespace(objects=objects)

def _normalize_orientation(ori: Sequence[int]) -> Tuple[int, int]:
    """Normalize orientation to one of four cardinal directions."""
    cand = (int(ori[0]), int(ori[1]))
    if cand in _ORI_TO_DEG:
        return cand
    
    vec = np.array(cand, dtype=float)
    if np.allclose(vec, 0.0):
        return (0, 1)
    
    basis = np.array(list(_ORI_TO_DEG.keys()), dtype=float)
    idx = int(np.argmax(basis @ vec))
    return tuple(int(x) for x in basis[idx])

def _rotate_orientation(ori: Tuple[int, int], degrees: int) -> Tuple[int, int]:
    """Rotate orientation by given degrees (must be multiple of 90)."""
    if int(degrees) % 90 != 0:
        raise ValueError(f"Invalid rotation: {degrees}")
    
    cur_deg = _ORI_TO_DEG.get(_normalize_orientation(ori), 0)
    new_deg = (cur_deg + int(degrees)) % 360
    return _DEG_TO_ORI.get(new_deg, (0, 1))

def _is_visible_from(
    agent_pos: Tuple[float, float],
    agent_ori: Tuple[int, int],
    target_pos: Tuple[float, float],
    fov: int = 90,
) -> bool:
    """Check if target is visible from agent position and orientation."""
    if agent_pos == target_pos:
        return False
    
    direction = np.array(target_pos, dtype=float) - np.array(agent_pos, dtype=float)
    if np.allclose(direction, 0.0):
        return False
    direction /= np.linalg.norm(direction)
    
    ori_vec = np.array(agent_ori, dtype=float)
    if np.allclose(ori_vec, 0.0):
        return False
    ori_vec /= np.linalg.norm(ori_vec)
    
    limit = 0.0 if int(fov) == 180 else _COS_45
    return float(np.dot(direction, ori_vec)) >= limit - 1e-3


# ========== Navigation Simulation ==========
def _simulate_navigation(
    actions: List[Tuple[str, Any]],
    init_pos: Tuple[float, float],
    init_ori: Tuple[int, int],
    object_positions: Dict[str, Tuple[float, float]],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]], Optional[str]]:
    """Simulate navigation actions and return final position, orientation, and any error."""
    pos = _coerce_point(init_pos)
    ori = _normalize_orientation(init_ori)
    
    for action_type, param in actions:
        if action_type == 'rotate':
            try:
                ori = _rotate_orientation(ori, int(param))
            except ValueError:
                return None, None, 'invalid_rotation'
        
        elif action_type == 'jumpto':
            name = str(param).lower()
            target = object_positions.get(name)
            if target is None:
                return None, None, 'unknown_object'
            if not _is_visible_from(pos, ori, target):
                return None, None, 'target_not_visible'
            pos = target
        
        else:
            return None, None, 'invalid_action'
    
    return pos, ori, None


# ========== Task-Specific Evaluators ==========
def _eval_token_sequence(pred: str, answer: Sequence[str]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate exact token sequence match."""
    tokens = _extract_sequence(pred, str)
    if not tokens or len(tokens) != len(answer):
        return False, {}
    
    expected = [_normalize_whitespace(item) for item in answer]
    actual = [_normalize_whitespace(token) for token in tokens]
    return actual == expected, {}

def _eval_rotation_direction(pred: str, answer: str) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate clockwise/counterclockwise."""
    options = {
        'clockwise': {'clockwise', 'cw'},
        'counterclockwise': {'counterclockwise', 'counter-clockwise', 'anticlockwise', 'anti-clockwise', 'ccw'},
    }
    
    predicted = _normalize_joined(pred)
    target = _normalize_joined(answer)
    key = 'counterclockwise' if target in {'counterclockwise', 'anticlockwise', 'ccw'} else 'clockwise'
    synonyms = {_normalize_joined(item) for item in options[key]}
    
    return any(syn in predicted for syn in synonyms), {}

def _eval_multiple_choice(pred: str, answer: str, choices: Optional[List[str]]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate multiple choice answer."""
    guess = _require_text(pred)
    if not guess:
        return False, {}
    
    if choices:
        labels = [chr(65 + idx) for idx in range(len(choices))]
        if guess.upper() in labels:
            return guess.upper() == str(answer).strip().upper(), {}
    
    return _labels_match(guess, answer), {}

def _eval_direction_text(pred: str, answer: Union[str, Sequence[str]]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate direction and distance pair."""
    parsed = _parse_direction_distance(pred)
    
    # Normalize answer
    normalized_answer = None
    if isinstance(answer, (list, tuple)) and len(answer) >= 2:
        normalized_answer = (_canonicalize_label(str(answer[0])), _canonicalize_label(str(answer[1])))
    else:
        parsed_ans = _parse_direction_distance(str(answer))
        if parsed_ans:
            normalized_answer = (_canonicalize_label(parsed_ans[0]), _canonicalize_label(parsed_ans[1]))
    
    if not parsed or not normalized_answer:
        return False, {}
    
    # Canonicalize parsed prediction
    normalized_pred = (_canonicalize_label(parsed[0]), _canonicalize_label(parsed[1]))
    
    return (_labels_match(normalized_pred[0], normalized_answer[0]) + _labels_match(normalized_pred[1], normalized_answer[1])) / 2, {}

def _eval_coordinate_list(pred: str, answer: Sequence[Tuple[int, int]]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate list of coordinates."""
    coords = _parse_coordinate_list(pred)
    if coords is None or len(coords) != len(answer):
        return False, {}
    
    try:
        expected = [_coerce_point(pair) for pair in answer]
    except ValueError:
        return False, {}
    
    ok = all(_points_close(c, e) for c, e in zip(coords, expected))
    return ok, {}

def _eval_exact_text(pred: str, answer: Union[str, Sequence[str]]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate exact text match with label matching."""
    return _labels_match(pred, answer), {}

def _eval_forward_nav(pred: str, answer: str) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate forward navigation/localization task."""
    pred_rels = _parse_object_relations(pred)
    ans_rels = _parse_object_relations(answer)
    
    if not pred_rels or not ans_rels:
        return False, {}
    
    pred_dict = {obj: (_canonicalize_label(dir_), _canonicalize_label(dist)) 
                 for obj, dir_, dist in pred_rels}
    ans_dict = {obj: (_canonicalize_label(dir_), _canonicalize_label(dist)) 
                for obj, dir_, dist in ans_rels}
    
    score = 0
    assert len(ans_dict) == 1, "Ground truth must have exactly one object"
    for obj in ans_dict:
        if obj in pred_dict:
            pred_dir, pred_dist = pred_dict[obj]
            ans_dir, ans_dist = ans_dict[obj]
            dir_match = _labels_match(pred_dir, ans_dir)
            dist_match = _labels_match(pred_dist, ans_dist)
            score = (dir_match + dist_match) / 2
    
    return score, {}

def _eval_backward_nav(pred: str, answer: Union[str, Dict]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate backward navigation task."""
    if not isinstance(answer, dict):
        return False, {}
    
    pred_actions = _parse_action_sequence(pred)
    if not pred_actions:
        return False, {'error': 'invalid_format'}
    
    # Extract ground truth data
    init_pos = _coerce_point(answer['init_pos'])
    init_ori = _normalize_orientation(answer['init_ori'])
    object_positions = {str(k).lower(): _coerce_point(v) for k, v in answer['object_positions'].items()}
    
    # Simulate navigation
    final_pos, final_ori, error = _simulate_navigation(pred_actions, init_pos, init_ori, object_positions)
    if error:
        return False, {'error': error}
    
    # Get expected observations
    final_visible = answer.get('final_observation', [])
    if not final_visible:
        return False, {'error': 'missing_ground_truth'}
    
    visibles: Dict[str, Tuple[str, str]] = {}
    for item in final_visible:
        name = str(item.get('name', '')).strip().lower()
        direction = _canonicalize_label(item.get('direction', ''))
        distance = _canonicalize_label(item.get('distance', ''))
        if name and direction and distance:
            visibles[name] = (direction, distance)
    
    def _relations_match(agent_pos: Tuple[float, float], agent_ori: Tuple[int, int]) -> bool:
        """Check if all visible objects have matching relations."""
        for obj_name, (dir_gt, dist_gt) in visibles.items():
            target = object_positions.get(obj_name)
            if target is None or not _is_visible_from(agent_pos, agent_ori, target):
                return False
            
            rel = PairwiseRelationshipDiscrete.relationship(target, agent_pos, anchor_ori=agent_ori)
            dir_label = _canonicalize_label(rel.direction.bin_label)
            dist_label = _canonicalize_label(rel.dist.bin_label)
            
            if dir_label != dir_gt or dist_label != dist_gt:
                return False
        return True
    
    # Check if predicted final state matches observations
    best_info = {'pos_match': False, 'ori_match': False, 'visible_match': False}
    if final_pos and final_ori and _relations_match(final_pos, final_ori):
        best_info.update({
            'pos_match': final_pos == tuple(answer['final_pos']),
            'ori_match': final_ori == tuple(answer['final_ori']),
            'visible_match': True,
        })
        return True, best_info
    
    # Also check ground truth position as fallback
    expected_pos = tuple(answer['final_pos'])
    expected_ori = tuple(answer['final_ori'])
    if _relations_match(expected_pos, expected_ori):
        best_info.update({'pos_match': True, 'ori_match': True})
    
    return False, best_info

def _calculate_coord_similarity(pred_coord: tuple[float, float], gt_coord: tuple[float, float]) -> float:
    pred = np.array(pred_coord, dtype=float)
    gt = np.array(gt_coord, dtype=float)

    rmse = np.linalg.norm(pred - gt)
    L = np.linalg.norm(gt)

    if L == 0:
        return 1.0 if rmse == 0 else 0.0
    similarity_score = np.exp(-rmse / L)

    return similarity_score

def _score_similarity_mra_style(similarity: float) -> float:
    similarity = max(0, min(1, similarity))
    
    quality_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70, 
        0.75, 0.80, 0.85, 0.90, 0.95
    ]
    
    passed_levels = 0
    for threshold in quality_thresholds:
        if similarity >= threshold:
            passed_levels += 1
            
    final_score = passed_levels / len(quality_thresholds)
    
    return final_score

def _eval_backward_loc(pred: str, answer: Any) -> Tuple[float, Dict[str, Any]]:
    """Evaluate backward localization task."""
    if not isinstance(answer, dict):
        return False, {}
    
    parsed = _parse_coord_orientation(pred)
    if not parsed:
        return False, {}
    
    coord_pred, ori_pred = parsed
    
    try:
        coord_ans = _coerce_point(answer.get('coord', ()))
    except ValueError:
        return False, {}
    
    ori_ans = str(answer.get('orientation', '')).strip().lower()
    similarity = _calculate_coord_similarity(coord_pred, coord_ans)
    score = _score_similarity_mra_style(similarity)
    ori_ok = ori_pred == ori_ans

    return (score + ori_ok) / 2, {
        'orientation_match': ori_ok,
        'similarity': similarity,
    }


# ========== E2A Evaluation ==========
def e2a_eval_fn(pred: Any, answer: Any) -> Tuple[float, Dict[str, Any]]:
    """Evaluate E2A (Egocentric to Allocentric) coordinate prediction.
    
    Answer can be:
    - List/tuple of coordinates
    - Dict with:
        - 'coords': ground truth coordinates (required)
        - 'absolute_coords': absolute positions of selected objects for normalization (optional)
        - 'threshold': similarity threshold (optional, default 0.9)
    """
    # Import here to avoid circular dependency
    from .cogmap.metrics import compute_pos_sim, compute_dir_sim
    
    if not isinstance(pred, str):
        return False, {'error': 'prediction_not_string'}
    
    coords = _parse_coordinate_list(pred)
    if not coords:
        return False, {'error': 'invalid_prediction_format'}
    
    # Extract ground truth and threshold
    threshold = _E2A_DEFAULT_SIM_THRESHOLD
    gt_coords_raw: Any = answer
    norm_coords_raw: Any = None  # Coordinates to use for normalization
    
    if isinstance(answer, dict):
        threshold = float(answer.get('threshold', threshold))
        gt_coords_raw = answer.get('coords', [])
        # Use absolute coordinates for normalization if provided
        norm_coords_raw = answer.get('absolute_coords', None)
    
    if not isinstance(gt_coords_raw, (list, tuple)) or not gt_coords_raw:
        return False, {'error': 'missing_ground_truth'}
    
    try:
        gt_coords = [_coerce_point(pt) for pt in gt_coords_raw]
    except ValueError:
        return False, {'error': 'invalid_ground_truth'}
    
    if len(coords) != len(gt_coords):
        return False, {'error': 'mismatched_coordinate_count'}
    
    # Use absolute coordinates for normalization if provided, otherwise use gt_coords
    if norm_coords_raw:
        try:
            norm_coords = [_coerce_point(pt) for pt in norm_coords_raw]
        except ValueError:
            norm_coords = gt_coords  # Fallback to gt_coords
    else:
        norm_coords = gt_coords
    
    pred_room = _coords_to_room(coords)
    gt_room = _coords_to_room(gt_coords)
    pos_norm_L = _coord_norm_from_gt(norm_coords)
    
    pos_sim = compute_pos_sim(pred_room, gt_room, allow_scale=False, pos_norm_L=pos_norm_L)
    dir_sim = compute_dir_sim(pred_room, gt_room)
    similarity = (pos_sim + dir_sim) / 2
    return _score_similarity_mra_style(similarity), {'similarity': similarity, 'threshold': threshold}


# ========== Public Helper Functions ==========
def exp_evaluate_fn(pred: str, relationships_to_query: List[Dict]) -> bool:
    """Evaluate exploration task (query object pairs or terminate)."""
    text = _require_text(pred)
    if text is None:
        return False
    
    if not relationships_to_query:
        return "terminate" in text.lower()
    
    pred_pair = extract_elements(text, expected_type=str)
    if not pred_pair or len(pred_pair) != 2:
        return False
    
    a, b = (_casefold(pred_pair[0]), _casefold(pred_pair[1]))
    for rel in relationships_to_query:
        x = _casefold(rel['object1'])
        y = _casefold(rel['object2'])
        if (a == x and b == y) or (a == y and b == x):
            return True
    
    return False

def _match_text_sequence(pred: str, expected: Sequence[Any]) -> bool:
    """Check if predicted sequence matches expected sequence."""
    if not pred or not isinstance(pred, str):
        return False
    
    items = extract_elements(pred, str)
    if not items or len(items) != len(expected):
        return False
    
    expected_lower = [_casefold(item) for item in expected]
    return all(_casefold(pred_item) == exp for pred_item, exp in zip(items, expected_lower))

def tuple_eval_fn(pred: str, ground_truth: tuple) -> bool:
    """Evaluate tuple/sequence prediction."""
    return _match_text_sequence(pred, ground_truth)

def list_dir_eval_fn(pred: str, gt_list: List[tuple]) -> bool:
    """Evaluate numbered list of directions."""
    pairs = _parse_indexed_lines(pred)
    if not pairs:
        return False
    
    correct = 0
    for idx, pred_dir in pairs:
        if 0 <= idx < len(gt_list) and _match_text_sequence(pred_dir, gt_list[idx]):
            correct += 1
    
    return correct == len(gt_list)

def obj_seq_eval_fn(pred: str, gt_object_sequence: List[str]) -> bool:
    """Evaluate object sequence prediction."""
    return _match_text_sequence(pred, gt_object_sequence)

def deg_seq_eval_fn(pred: str, gt_degree_sequence: List[int]) -> bool:
    """Evaluate degree sequence prediction."""
    if not pred or not isinstance(pred, str):
        return False
    
    pred_degrees = extract_elements(pred, int, clean_pattern=r'[°degrees\s]')
    return bool(pred_degrees and pred_degrees == list(gt_degree_sequence))

def obj_presence_eval_fn(pred: Any, answer: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate object presence detection with precision/recall metrics."""
    if not isinstance(pred, str):
        return False, {}
    
    pred_objects = extract_elements(pred, str)
    if not pred_objects:
        return False, {}
    
    gt_objects = {_casefold(obj) for obj in answer}
    pred_set = {_casefold(obj) for obj in pred_objects}
    
    correct_count = len(gt_objects & pred_set)
    total_gt = len(gt_objects)
    precision = correct_count / len(pred_set) if pred_set else 0.0
    recall = correct_count / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    info = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_count": correct_count,
        "total_gt": total_gt,
        "predicted_objects": sorted(pred_set),
        "ground_truth_objects": sorted(gt_objects),
    }
    
    return correct_count == total_gt, info

def multi_choice_eval_fn(pred: str, answer: Union[List[str], str]) -> bool:
    """Evaluate multiple choice with multiple acceptable answers."""
    text = _require_text(pred)
    if text is None:
        return False
    
    answers = [answer] if isinstance(answer, str) else answer
    return text.lower() in {_casefold(ans) for ans in answers}


# ========== Task Evaluator Registry ==========
TaskEvaluator = Callable[[Any, Any, Optional[List[str]]], Tuple[bool, Dict[str, Any]]]

def _wrap_eval(
    func: Callable,
    *,
    pred_cast: Callable[[Any], Any] | None = str,
    answer_cast: Callable[[Any], Any] | None = None,
    pass_choices: bool = False,
) -> TaskEvaluator:
    """Wrap evaluation function to handle type casting."""
    def wrapped(pred: Any, answer: Any, choices: Optional[List[str]]) -> Tuple[bool, Dict[str, Any]]:
        pred_arg = pred_cast(pred) if pred_cast else pred
        ans_arg = answer_cast(answer) if answer_cast else answer
        
        if pass_choices:
            return func(pred_arg, ans_arg, choices)
        return func(pred_arg, ans_arg)
    
    return wrapped

TASK_EVALUATORS: Dict[str, TaskEvaluator] = {
    'RotEvaluationTask': _wrap_eval(_eval_token_sequence),
    'RotDualEvaluationTask': _wrap_eval(_eval_rotation_direction),
    'DirectionEvaluationTask': _wrap_eval(_eval_direction_text),
    'PovEvaluationTask': _wrap_eval(_eval_direction_text),
    'BackwardPovEvaluationTask': _wrap_eval(_eval_exact_text),
    'DirectionPov': _wrap_eval(_eval_direction_text),
    'E2AEvaluationTask': lambda pred, answer, _choices: e2a_eval_fn(pred, answer),
    'ForwardFOVEvaluationTask': _wrap_eval(_eval_forward_nav),
    'BackwardNavEvaluationTask': _wrap_eval(_eval_backward_nav, pred_cast=str, answer_cast=None),
    'ForwardLocEvaluationTask': _wrap_eval(_eval_forward_nav),
    'BackwardLocEvaluationTask': _wrap_eval(_eval_backward_loc, pred_cast=str, answer_cast=None),
    'FalseBeliefDirectionPov': _wrap_eval(_eval_direction_text),
}

def evaluate_task_answer(
    task_type: str,
    pred: Any,
    answer: Any,
    choices: Optional[List[str]],
) -> Tuple[float, Dict[str, Any]]:
    """Main entry point for task evaluation."""
    evaluator = TASK_EVALUATORS.get(task_type)
    assert evaluator is not None, f"Unknown task evaluator: {task_type}"
    return evaluator(pred, answer, choices)
    
    # Fallback to multiple choice evaluation
    # if choices:
    #     return _eval_multiple_choice(str(pred), str(answer), choices)
    # return _eval_multiple_choice(str(pred), str(answer), None)


# ========== Exports ==========
__all__ = [
    'TaskEvaluator',
    'TASK_EVALUATORS',
    'evaluate_task_answer',
    'exp_evaluate_fn',
    'tuple_eval_fn',
    'list_dir_eval_fn',
    'obj_seq_eval_fn',
    'deg_seq_eval_fn',
    'obj_presence_eval_fn',
    'multi_choice_eval_fn',
    'e2a_eval_fn',
    'extract_elements',
]
