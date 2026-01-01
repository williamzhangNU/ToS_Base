"""
Cognitive map prompts (modular, per-type).

A single BASE prompt contains all shared schema and general rules.
Per-type prompts ONLY add their specific instructions (no repetition).
"""

from typing import Optional, Dict, List, Tuple
from ..utils.room_utils import RoomPlotter

BASE_COGMAP_PROMPT = """\
## Cognitive Map (JSON)

Represent the scene as a JSON map. 

### Schema (shared)
- position: [x, y] integers
- facing: "north|south|east|west" (global) or "+x|-x|+y|-y" (local/rooms)

### General rules (shared)
- Include only observed objects.
- MUST include facing key if the object has facing direction.
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
## Global Cognitive Map 

- Grid: concise global map on an N×M grid.
- Frame: origin [0,0] is your initial position; your initial facing direction is north.
- Content: include all observed objects and gates; include the agent
- Facing: use "north|south|east|west" (cardinal direction only).
- Other candidates: a JSON list of up to 3 alternative positions, e.g. [[x1,y1],[x2,y2]].
  - MUST be valid positions for the object. Exclude position in `position`.
  - If the position is unique (no ambiguity), output [].

Example:
```json
{
    "agent": {"position": [2, 3], "facing": "east", "other_candidates": []},
    "chair": {"position": [2, 4], "facing": "north", "other_candidates": []},
    "sofa": {"position": [5, 1], "facing": "west", "other_candidates": [[4, 1], [5, 2]]}
}
```
"""

# Local-only specifics
COGMAP_INSTRUCTION_LOCAL_ONLY = """\
## Local Cognitive Map

- Structure: include an "objects" dict; each object's position and facing are relative to the agent at time of writing.
- Frame: must include "origin":"agent". Always keep in mind that the origin is the agent's current position and orientation.
  - +y: facing forward
  - when facing +y: +x -> right, -x -> left, -y -> backward
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

def get_cogmap_prompt(map_type: str, enable_think: bool = True, all_candidate_coords: Optional[List[Tuple[int, int]]] = None, use_vision=False, room=None, agent=None) -> str:
    """Return the assembled cognitive-map prompt for a given type, with format rules."""
    t = (map_type or "global").strip().lower()
    fmt = _cogmap_format_rules(enable_think)
    if t == "global":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n{fmt}"
    if t == "local":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_LOCAL_ONLY}\n\n{fmt}"
    if t == "unexplored":
        return get_unexplored_prompt(enable_think, all_candidate_coords)
    if t == "fog_probe":
        return get_fog_probe_prompt(enable_think, use_vision, room, agent, all_candidate_coords)
    # default to global
    return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n{fmt}"

# --- Fog Probe Prompt ---
FOG_PROBE_INSTRUCTION = """\
### Fog Probe
{symbol_def}

The map displays candidate points labeled A-Z.
Select the points that are located in unexplored/unobserved regions.

Map:
{symbol_map}
Example:
```json
{{
    "unexplored": ["A", "C"]
}}
```
"""

def get_fog_probe_prompt(enable_think, use_vision, room, agent, all_candidate_coords) -> str:
    symbol_def = "" if use_vision else RoomPlotter.get_symbol_definition()
    symbol_map = "<image>" if use_vision else RoomPlotter.get_symbolic_map(room, agent, False, False, all_candidate_coords)
    instruction = FOG_PROBE_INSTRUCTION.format(symbol_def=symbol_def, symbol_map=symbol_map)
    fmt = _cogmap_format_rules(enable_think)
    return f"{instruction}\n\n{fmt}"

# --- Unexplored Areas Prompt ---
UNEXPLORED_INSTRUCTION = """\
### Unexplored Areas
Select ALL coordinates that correspond to points in **unexplored/unobserved** regions.
{candidate_coords_str}
### Rules
- Coordinates are in GLOBAL coordinates (relative to your starting position ([0,0]) and orientation (north))
- Output only the selected coordinates in the same format
- Separate with semicolons

Example:
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
