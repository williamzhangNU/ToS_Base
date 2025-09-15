"""
Cognitive map prompts (modular, per-type).

Keep prompts concise. Exposed via get_cogmap_prompt(map_type).
"""

# Global-only
COGMAP_INSTRUCTION_GLOBAL_ONLY = """\
## Cognitive Map (global only)

Keep a concise global map JSON of the scene on a N by M grid.

- Global: origin [0,0] and +Y is your initial facing direction

Fields:
- position: [x, y] in the map’s coordinate system (integers or integer-like)
- facing: one of "north|south|east|west" (omit or set unknown if not applicable)
- confidence: "high" (certain), "medium" (estimated), "low" (unknown)

Content rules:
- Global: include observed objects and gates; include agent; exclude "initial_pos"
Always output the cognitive map JSON first in your thinking.
"""

# Local-only
COGMAP_INSTRUCTION_LOCAL_ONLY = """\
## Cognitive Map (local only)

Keep a concise local map JSON.

- Local: must include "origin":"agent" and an "objects" dict. Each object's
  position and facing are relative to the agent at the time of writing.
- Facings use +x, -x, +y, -y relative to the local frame.
- Include only currently visible objects; exclude the agent itself here.
"""

# Rooms-only
COGMAP_INSTRUCTION_ROOMS_ONLY = """\
## Cognitive Map (rooms only)

Keep concise per-room maps.

- Each room entry is keyed by room id and must include "origin":"<gate_name>" and an "objects" dict.
- The origin gate is the gate first used to enter that room. +Y points into the room.
- Do not include the agent or origin gate inside a room's objects.
- Exclude the initial room if it has no origin gate.
"""

def get_cogmap_prompt(map_type: str) -> str:
    mt = (map_type or "global").lower()
    if mt == "global":
        return COGMAP_INSTRUCTION_GLOBAL_ONLY
    if mt == "local":
        return COGMAP_INSTRUCTION_LOCAL_ONLY
    if mt == "rooms":
        return COGMAP_INSTRUCTION_ROOMS_ONLY
    # default to global
    return COGMAP_INSTRUCTION_GLOBAL_ONLY


