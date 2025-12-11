"""
Cognitive map prompts (modular, per-type).

A single BASE prompt contains all shared schema and general rules.
Per-type prompts ONLY add their specific instructions (no repetition).
"""

from typing import Optional, Dict, List, Tuple

BASE_COGMAP_PROMPT = """\
## Cognitive Map (JSON)

Represent the scene as a JSON map. 

### Schema (shared)
- position: [x, y] integers (or integer-like)
- facing: "north|south|east|west" (global) or "+x|-x|+y|-y" (local/rooms)

### General rules (shared)
- Include only observed objects.
- Approximate positions to nearest integer.
- MUST include facing key if the object has facing direction according to the label image that has two copies.
"""

from ..utils.utils import THINK_LABEL, ANSWER_LABEL

def _cogmap_format_rules(enable_think: bool) -> str:
    if enable_think:
        return (
            "!!! IMPORTANT OUTPUT RULES !!!\n"
            f"1. Always output (labels followed by a newline):\n{THINK_LABEL} [Your thoughts on cognitive map]\n{ANSWER_LABEL} [JSON map only]\n"
            f"2. Inside {ANSWER_LABEL} output ONLY the JSON (no prose).\n"
            "3. Any deviation is invalid."
        )
    return (
        "!!! IMPORTANT OUTPUT RULES !!!\n"
        f"1. Always output (label followed by a newline):\n{ANSWER_LABEL} [JSON map only]\n"
        f"2. Inside {ANSWER_LABEL} output ONLY the JSON (no prose).\n"
        "3. Any deviation is invalid."
    )

# Global-only specifics
COGMAP_INSTRUCTION_GLOBAL_ONLY = """\
## Cognitive Map — Global (specifics)

- Grid: concise global map on an N×M grid.
- Frame: origin [0,0] is your initial position; your initial facing direction is north.
- Content: include all observed objects and gates; include the agent
- Facing: use "north|south|east|west".
- Confidence: use "high" if you are certain about the position prediction, "low" if uncertain.

Example:
```json
{
    "agent": {"position": [2, 3], "facing": "east", "confidence": "high"},
    "chair": {"position": [2, 4], "facing": "north", "confidence": "low"},
}
```
"""

# Local-only specifics
COGMAP_INSTRUCTION_LOCAL_ONLY = """\
## Cognitive Map — Local (specifics)

- Structure: include an "objects" dict; each object's position and facing are relative to the agent at time of writing.
- Frame: must include "origin":"agent". Always keep in mind that the origin is the agent's current position and orientation.
  - +y: facing forward
  - +x: right when facing +y; -x: left; -y: back toward door
  - All positions/facings relative to this frame.
- Content: "objects" dict; include all objects and doors in your current field of view; exclude agent.
- Facing: use "+x|-x|+y|-y" (local axes).

Example:
```json
{
    "origin": "agent",
    "objects": {
      "chair": {"position": [0, 1], "facing": "-x"}
    } 
}
```
"""

# Rooms-only specifics
COGMAP_INSTRUCTION_ROOMS_ONLY = """\
## Cognitive Map — Rooms (specifics)

- Structure: rooms keyed by room id.
- Frame per room: include "origin":"<door_name>|initial".
  - origin is the first entry door; for starting room use "initial" (starting position)
  - +y: into room (direction of walking through the door); north for starting room
  - +x: right when facing +y; -x: left; -y: back toward door
  - All positions/facings relative to this frame.
- Content: room's "objects" dict; exclude agent and entry door.
- Facing: "+x|-x|+y|-y".

Example:
```json
{
  "1": {
    "origin": "initial",
    "objects": {
      "chair": {"position": [1, 0], "facing": "+y"},
    }
  },
  "2": {
    "origin": "red door",
    "objects": {
      "sofa": {"position": [0, 2], "facing": "-y"}
    }
  }
}
```
"""

def get_cogmap_prompt(map_type: str, enable_think: bool = True, all_candidate_coords: Optional[List[Tuple[int, int]]] = None) -> str:
    """Return the assembled cognitive-map prompt for a given type, with format rules."""
    t = (map_type or "global").strip().lower()
    fmt = _cogmap_format_rules(enable_think)
    if t == "global":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n{fmt}"
    if t == "local":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_LOCAL_ONLY}\n\n{fmt}"
    if t == "rooms":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_ROOMS_ONLY}\n\n{fmt}"
    if t == "relations":
        return f"{RELATIONS_PROMPT}\n\n{fmt}"
    if t == "unexplored":
        return get_unexplored_prompt(enable_think, all_candidate_coords)
    if t == "false_belief":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n{fmt}"
    # default to global
    return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n{fmt}"

# --- Unexplored Areas Prompt ---
UNEXPLORED_INSTRUCTION = """\
### Task
Select ALL coordinates that correspond to points in unexplored regions.
{candidate_coords_str}
### Rules
- Coordinates are in GLOBAL coordinates (relative to your initial position and orientation)
- Output only the selected coordinates in the same format
- Separate multiple coordinates with semicolons
- If all areas are explored, output "none"

### Output Format
Just the coordinates separated by semicolons:
```json
{{
    "unexplored": "(5, 3); (2, 1); (10, 2)"
}}
```
"""

def get_unexplored_prompt(enable_think: bool = True, all_candidate_coords: Optional[List[Tuple[int, int]]] = None) -> str:
    """Return the unexplored areas prompt with format rules.
    
    Args:
        enable_think: Whether to include thinking section
        all_candidate_coords: List of all candidate (x, y) coordinates
    """
    # Format all coordinates
    if all_candidate_coords:
        coord_strs = [f"({x}, {y})" for x, y in all_candidate_coords]
        candidate_coords_str = '; '.join(coord_strs)
    else:
        raise ValueError("all_candidate_coords must be provided for unexplored prompt.")
    
    instruction = UNEXPLORED_INSTRUCTION.format(candidate_coords_str=candidate_coords_str)
    fmt = _cogmap_format_rules(enable_think)
    return f"{instruction}\n\n{fmt}"


# --- Pairwise relations ---
from ..utils.relation_codes import _DIR_LABEL_TO_CODE as _DLC, _DIST_LABEL_TO_CODE as _SLC

def _build_relations_mapping_text() -> str:
    dir_pairs = ", ".join([f"{lab}={code}" for lab, code in _DLC.items()])
    dist_pairs = ", ".join([f"{lab}={code}" for lab, code in _SLC.items()])
    return f"Directions: {dir_pairs}. Distances: {dist_pairs}."

RELATIONS_PROMPT = f"""\
## Pairwise Relations (JSON)

- Report unordered pairs for ALL observed objects (objects and gates) and agent's initial position as "initial". Do NOT include agent current pose.
- Keys: "A|B" (alphabetical). A|B means A is relative to B.
- Report only one direction per pair (A|B or B|A, not both)
- Values: "(DIR, DIST)" where DIR and DIST are compact codes. {_build_relations_mapping_text()}
- Output a flat JSON object of pairs (no extra nesting).

Example:
```json
{{
  "initial|chair": "(E, near)",
  "chair|door1": "(NW, mid)"
}}
```
"""