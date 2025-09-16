from typing import Tuple


# Compact codes for CardinalBinsAllo labels
_DIR_LABEL_TO_CODE = {
    "north": "N",
    "north east": "NE",
    "east": "E",
    "south east": "SE",
    "south": "S",
    "south west": "SW",
    "west": "W",
    "north west": "NW",
}
_DIR_CODE_TO_LABEL = {v: k for k, v in _DIR_LABEL_TO_CODE.items()}

# Compact codes for StandardDistanceBins labels
_DIST_LABEL_TO_CODE = {
    "same distance": "same",
    "near": "near",
    "mid distance": "mid",
    "slightly far": "sfar",
    "far": "far",
    "very far": "vfar",
    "extremely far": "xfar",
}
_DIST_CODE_TO_LABEL = {v: k for k, v in _DIST_LABEL_TO_CODE.items()}


def _normalize_key(s: str) -> str:
    return str(s).strip().lower()


def direction_label_to_code(label: str) -> str:
    return _DIR_LABEL_TO_CODE.get(_normalize_key(label), _normalize_key(label))


def direction_code_to_label(code: str) -> str:
    return _DIR_CODE_TO_LABEL.get(_normalize_key(code), _normalize_key(code))


def distance_label_to_code(label: str) -> str:
    return _DIST_LABEL_TO_CODE.get(_normalize_key(label), _normalize_key(label))


def distance_code_to_label(code: str) -> str:
    return _DIST_CODE_TO_LABEL.get(_normalize_key(code), _normalize_key(code))


def to_code(value: str) -> str:
    """Unified: map direction/distance label-or-code to code.
    Examples: 'north'->'n'; 'n'->'n'; 'mid distance'->'mid'; 'mid'->'mid'.
    """
    v = _normalize_key(value)
    v_upper = v.upper()
    # direction codes/labels → uppercase codes
    if v in _DIR_LABEL_TO_CODE:
        return _DIR_LABEL_TO_CODE[v]
    # accept lower/upper direction codes
    if v_upper in _DIR_CODE_TO_LABEL:
        return v_upper
    if v in _DIST_CODE_TO_LABEL:
        return v
    if v in _DIST_LABEL_TO_CODE:
        return _DIST_LABEL_TO_CODE[v]
    return v


def to_label(value: str) -> str:
    """Unified: map direction/distance code-or-label to label.
    Examples: 'n'->'north'; 'north'->'north'; 'mid'->'mid distance'.
    """
    v = _normalize_key(value)
    if v.upper() in _DIR_CODE_TO_LABEL:
        return _DIR_CODE_TO_LABEL[v.upper()]
    if v in _DIST_CODE_TO_LABEL:
        return _DIST_CODE_TO_LABEL[v]
    if v in _DIR_LABEL_TO_CODE:
        return v
    if v in _DIST_LABEL_TO_CODE:
        return v
    return v


def encode_relation_codes(dir_value: str, dist_value: str) -> str:
    """Encode direction/distance values (code or label) to short tuple string '(dir, dist)'."""
    d = to_code(dir_value)
    r = to_code(dist_value)
    return f"({d}, {r})"


def decode_relation_codes(text: str) -> Tuple[str, str]:
    """Decode '(nw, mid)' or 'nw, mid' to (dir_code, dist_code)."""
    s = str(text).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        return "", ""
    return to_code(parts[0]), to_code(parts[1])


def make_pair_key(a: str, b: str) -> str:
    """Canonical unordered pair key 'A|B' with lexicographic order (case-insensitive)."""
    a_s, b_s = str(a), str(b)
    return (f"{a_s}|{b_s}" if a_s.lower() <= b_s.lower() else f"{b_s}|{a_s}")


