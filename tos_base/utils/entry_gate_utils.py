import re
from typing import Dict, Set, Any, Optional
from ..core.room import BaseRoom, Room

def extract_entry_map(proxy) -> dict[int, str]:
    """
    Build {room_id: first_gate_used_to_enter_it} using only `proxy`.

    Rules:
      - Prefill rooms (≠ start) that have exactly one gate in proxy.gates_by_room.
      - Treat any Move(door_k) as entering the room on the *other* side of that door.
      - Never assign an entry for the start room (room 1 by default, or agent.init_room_id).
      - First gate wins for a room (don't overwrite once set).
    """
    rooms_by_gate: dict = getattr(proxy.room, "rooms_by_gate", {})
    gates_by_room: dict = getattr(proxy, "gates_by_room", {})  
    if not rooms_by_gate or not gates_by_room:
        return None

    start_room = int(getattr(proxy.agent, "init_room_id", getattr(proxy.agent, "room_id", 1)))
    cur = start_room

    # --- expected rooms to fill (rooms that have at least one gate, excluding start) ---
    rooms_with_gates = {int(r) for r, gs in gates_by_room.items() if gs}
    targets = rooms_with_gates - {start_room}

    # --- prefill rooms that have *exactly one* gate (the only possible entry gate) ---
    entry: dict[int, str] = {}
    for rid, gate_set in gates_by_room.items():
        rid = int(rid)
        if rid == start_room:
            continue
        if len(gate_set) == 1 and rid in targets:
            entry[rid] = next(iter(gate_set))

    # --- early exit if prefill solved all ---
    if entry.keys() >= targets:
        return entry

    # --- walk the turns: record the *first* door actually used to enter each room ---
    known_gates = set(rooms_by_gate.keys())
    for turn in getattr(proxy, "turns", []) or []:
        for act in getattr(turn, "actions", []) or []:
            if getattr(act, "action_type", None) != "move":
                continue
            g = (getattr(act, "data", {}) or {}).get("target_name")
            if g not in known_gates:
                continue

            a, b = [int(x) for x in rooms_by_gate[g]]
            nxt = b if cur == a else a
            if nxt != cur and nxt != start_room and nxt in targets and nxt not in entry:
                entry[nxt] = g
                if entry.keys() >= targets:  # all done
                    return entry
            cur = nxt

    return entry
