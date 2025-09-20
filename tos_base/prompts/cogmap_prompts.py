"""
Cognitive map prompts (modular, per-type).

A single BASE prompt contains all shared schema and general rules.
Per-type prompts ONLY add their specific instructions (no repetition).
"""

BASE_COGMAP_PROMPT = """\
## Cognitive Map (JSON)

Represent the scene as a JSON map. 

### Schema (shared)
- position: [x, y] integers (or integer-like)
- facing: use cardinal words ("north|south|east|west") for global/rooms frames; use axis signs ("+x|-x|+y|-y") for local frames

### General rules (shared)
- Coordinate frame MUST be explicit in the content.
- Include only observed/known information—do not invent.

In your thinking (<think> ... </think>):  
1) Briefly reason about your cognitive map 
2) Then provide the cognitive map JSON in <answer>...</answer>
"""

# Global-only specifics
COGMAP_INSTRUCTION_GLOBAL_ONLY = """\
## Cognitive Map — Global (specifics)

- Grid: concise global map on an N×M grid.
- Frame: origin [0,0] is your initial position; +Y is your initial facing direction (north).
- Content: include all observed objects and gates; include the agent
- Facing: use "north|south|east|west".

Example:
```json
{
    "agent": {"position": [2, 3], "facing": "east"},
    "chair": {"position": [2, 4], "facing": "north"}
}
```
"""

# Local-only specifics
COGMAP_INSTRUCTION_LOCAL_ONLY = """\
## Cognitive Map — Local (specifics)

- Frame: must include "origin":"agent". Always keep in mind that the origin is the agent's position and orientation.
- Structure: include an "objects" dict; each object's position and facing are relative to the agent at time of writing.
- Content: include all objects and doors in your current field of view; DO NOT include the agent itself in "objects".
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

- Structure: map of rooms keyed by room id.
- Frame per room: must include "origin":"<gate_name>" where the origin gate is the first used to enter that room; +Y points into the room.
- Content: include each room’s "objects" dict; DO NOT include the agent; DO NOT include the entry door.
- Initial room: include it with origin at the initial position and orientation.
- Facing: use "+x|-x|+y|-y".

Example:
```json
{
  "1": {
    "origin": "initial_pos",
    "objects": {
      "chair": {"position": [1, 0], "facing": "+y"},
      "table": {"position": [2, 1], "facing": "-x"}
    }
  },
  "2": {
    "origin": "door",
    "objects": {
      "sofa": {"position": [0, 2], "facing": "-y"}
    }
  }
}
```
"""

def get_cogmap_prompt(map_type: str) -> str:
    """Return the assembled cognitive-map prompt for a given type."""
    t = (map_type or "global").strip().lower()
    if t == "global":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}"
    if t == "local":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_LOCAL_ONLY}"
    if t == "rooms":
        return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_ROOMS_ONLY}"
    if t == "relations":
        return RELATIONS_PROMPT
    # default to global
    return f"{BASE_COGMAP_PROMPT}\n\n{COGMAP_INSTRUCTION_GLOBAL_ONLY}"

# --- Pairwise relations ---
from ..utils.relation_codes import _DIR_LABEL_TO_CODE as _DLC, _DIST_LABEL_TO_CODE as _SLC

def _build_relations_mapping_text() -> str:
    dir_pairs = ", ".join([f"{lab}={code}" for lab, code in _DLC.items()])
    dist_pairs = ", ".join([f"{lab}={code}" for lab, code in _SLC.items()])
    return f"Directions: {dir_pairs}. Distances: {dist_pairs}."

RELATIONS_PROMPT = f"""\
## Pairwise Relations (JSON)

- Report unordered pairs for ALL observed objects (objects and gates) and agent's initial position as "initial_pos". Do NOT include agent current pose.
- Keys: "A|B" (alphabetical). A|B means A is relative to B.
- Values: "(DIR, DIST)" where DIR and DIST are compact codes. {_build_relations_mapping_text()}
- Output a flat JSON object of pairs (no extra nesting).

Example:
```json
{{
  "initial_pos|chair": "(E, near)",
  "chair|door1": "(NW, mid)"
}}
```
"""