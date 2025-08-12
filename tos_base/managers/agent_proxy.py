from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np

from ragen.env.spatial.Base.tos_base import (
    Room,
    BaseAction,
    MoveAction,
    RotateAction,
    ObserveAction,
    TermAction,
    ExplorationManager,
    Agent,
)
from ragen.env.spatial.Base.tos_base.actions import GoThroughDoorAction
from ..utils.action_utils import action_results_to_text


@dataclass
class Turn:
    actions: List
    pos: tuple
    ori: tuple


def _ori_to_deg(ori: np.ndarray) -> int:
    mapping = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
    return mapping[tuple(int(x) for x in ori.tolist())]


def _closest_cardinal(vec: np.ndarray) -> np.ndarray:
    basis = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
    dots = [float(np.dot(vec, b)) for b in basis]
    return basis[int(np.argmax(dots))]


class AgentProxy:
    """Proxy agent supporting simple exploration strategies across rooms."""

    def __init__(self, room: Room, agent: Agent, strategy: str = "oracle"):
        assert isinstance(room, Room), "AgentProxy requires Room"
        self.mgr = ExplorationManager(room, agent)
        self.room, self.agent, self.strategy = self.mgr.exploration_room, self.mgr.agent, strategy

        self.gates_by_room: Dict[int, Set[str]] = {int(r): set(glist) for r, glist in self.room.gates_by_room.items()}

        self.nodes_by_room: Dict[int, Set[str]] = {}
        for o in self.room.objects:
            self.nodes_by_room.setdefault(int(o.room_id), set()).add(o.name)
        self.known_nodes_by_room: Dict[int, Set[str]] = {rid: set() for rid in self.nodes_by_room}

        self.all_edges_by_room: Dict[int, Set[frozenset]] = {}
        for rid, objs in self.nodes_by_room.items():
            if not objs:
                self.all_edges_by_room[rid] = set()
                continue
            gates = self.gates_by_room.get(rid, set())
            pairs: Set[frozenset] = set()
            for a in objs | gates:
                for b in objs | gates:
                    if a != b:
                        pairs.add(frozenset({a, b}))
            self.all_edges_by_room[int(rid)] = pairs
        self.known_edges_by_room: Dict[int, Set[frozenset]] = {int(rid): set() for rid in self.all_edges_by_room}

        self.total_nodes: int = sum(len(v) for v in self.nodes_by_room.values())

        self.turns: List[Turn] = []
        self.visited: Set[int] = set()
        self.current_gate: str | None = None

    def _current_room(self) -> int:
        return int(self.agent.room_id)

    def _add_turn(self, actions: List) -> None:
        self.turns.append(
            Turn(
                actions=list(actions),
                pos=tuple(int(x) for x in self.agent.pos.tolist()),
                ori=tuple(int(x) for x in self.agent.ori.tolist()),
            )
        )

    def _update_known_from_observe(self, last_result) -> None:
        rid = self._current_room()
        for n in last_result.data.get("visible_objects", []):
            self.known_nodes_by_room.setdefault(rid, set()).add(n)

    def _rotate_to_face(self, target_pos: np.ndarray) -> List:
        vec = target_pos - self.agent.pos
        if np.allclose(vec, 0):
            return []
        desired = _closest_cardinal(vec)
        cur, des = _ori_to_deg(self.agent.ori), _ori_to_deg(desired)
        return self._rotate_by(des - cur)

    def _rotate_by(self, delta: int) -> List:
        delta = delta % 360
        return [self.mgr.execute_action(RotateAction(delta))] if delta else []

    def _move_to(self, name: str) -> List:
        target = self.room.get_object_by_name(name)
        acts = []
        acts += self._rotate_to_face(target.pos)
        acts.append(self.mgr.execute_action(MoveAction(name)))
        return acts

    def _observe(self, prefix_actions: List = None) -> None:
        acts = list(prefix_actions or [])
        obs = self.mgr.execute_action(ObserveAction())
        acts.append(obs)
        self._update_known_from_observe(obs)
        self._add_turn(acts)

    def _unknown_nodes_in_room(self, rid: int) -> Set[str]:
        return self.nodes_by_room.get(rid, set()) - self.known_nodes_by_room.get(rid, set())

    def _unknown_edges_in_room(self, rid: int) -> Set[frozenset]:
        return self.all_edges_by_room.get(rid, set()) - self.known_edges_by_room.get(rid, set())

    def _score_rotation(self, rot: int, target_names: Set[str]) -> int:
        if not target_names:
            return 0
        objects = {o.name: o for o in self.room.all_objects}
        R = BaseAction._get_rotation_matrix(rot)
        tmp_agent = self.agent.copy(); tmp_agent.ori = self.agent.ori @ R
        return sum(1 for n in target_names if n in objects and BaseAction._is_visible(tmp_agent, objects[n]))

    def _gate_between(self, a: int, b: int) -> str:
        for gate_name, rooms in self.room.rooms_by_gate.items():
            if set(rooms) == {int(a), int(b)}:
                return gate_name
        raise AssertionError(f"No gate between rooms {a} and {b}")

    def _traverse_to(self, next_rid: int) -> List:
        cur = self._current_room()
        gate_name = self._gate_between(cur, next_rid)
        gate_obj = self.room.get_object_by_name(gate_name)
        acts = []
        acts += self._rotate_to_face(gate_obj.pos)
        acts.append(self.mgr.execute_action(MoveAction(gate_name)))
        go = self.mgr.execute_action(GoThroughDoorAction(gate_name))
        acts.append(go)
        self.current_gate = gate_name
        return acts

    def _entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        fov = BaseAction.get_field_of_view()
        allowed = (0, 90, 180, 270) if is_initial else ((0,) if fov == 180 else (0, 90, 270))
        rid = self._current_room()

        def _mark_current_gate_edges_known():
            if self.strategy != 'inquisitor' or not self.current_gate:
                return
            last_obs = self.turns[-1].actions[-1]
            vis = last_obs.data.get('visible_objects', [])
            known = self.known_edges_by_room.setdefault(rid, set())
            for n in vis:
                known.add(frozenset({self.current_gate, n}))

        if self.strategy == 'oracle':
            while self._unknown_nodes_in_room(rid):
                targets = self._unknown_nodes_in_room(rid)
                scores = {d: self._score_rotation(d, targets) for d in allowed}
                best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
                if best_score == 0:
                    break
                self._observe((prefix_actions or []) + self._rotate_by(best))
                prefix_actions = []
            return

        for d in allowed:
            if self._all_nodes_known_globally():
                break
            self._observe((prefix_actions or []) + self._rotate_by(d))
            prefix_actions = []
            _mark_current_gate_edges_known()

    def _inquisitor_edges(self) -> None:
        rid = self._current_room()
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return
        anchors = sorted(self.nodes_by_room[rid] | self.gates_by_room[rid])
        for anchor in anchors:
            if not any(anchor in p for p in unknown):
                continue
            self._observe(self._move_to(anchor))
            for d in (0, 90, 180, 270):
                if d == 0:
                    continue
                self._observe(self._rotate_by(d))
                last_obs = self.turns[-1].actions[-1]
                for n in last_obs.data.get('visible_objects', []):
                    if n == anchor:
                        continue
                    e = frozenset({anchor, n})
                    self.known_edges_by_room.setdefault(rid, set()).add(e)
                    if e in unknown:
                        unknown.discard(e)
                if not any(anchor in p for p in unknown):
                    break
        assert unknown == set(), f"Unknown pairs not resolved: {unknown}"

    def _all_nodes_known_globally(self) -> bool:
        known_total = sum(len(v) for v in self.known_nodes_by_room.values())
        return known_total >= self.total_nodes

    def _subtree_has_nodes(self, start_rid: int) -> bool:
        stack = [int(start_rid)]
        seen = set(self.visited)
        while stack:
            r_id = int(stack.pop())
            if r_id in seen:
                continue
            seen.add(r_id)
            if len(self.nodes_by_room.get(r_id, set())) > 0:
                return True
            stack.extend(int(adj_r_id) for adj_r_id in self.room.adjacent_rooms_by_room.get(r_id, []))
        return False

    def _dfs(self, rid: int, is_initial: bool, pending_prefix: List = None) -> List:
        self.visited.add(rid)
        self._entry_observe(is_initial=is_initial, prefix_actions=pending_prefix or [])
        if self.strategy == 'inquisitor':
            self._inquisitor_edges()
        if self._all_nodes_known_globally():
            return []
        carry: List = []
        for nxt in sorted(self.room.adjacent_rooms_by_room.get(rid, [])):
            if nxt in self.visited:
                continue
            if self.strategy == 'oracle' and not self._subtree_has_nodes(int(nxt)):
                continue
            to_child = carry + self._traverse_to(int(nxt))
            carry = self._dfs(int(nxt), is_initial=False, pending_prefix=to_child)
            carry = carry + self._traverse_to(int(rid))
        return carry

    def run(self) -> List[Turn]:
        start_rid = self._current_room()
        pending = self._dfs(start_rid, is_initial=True, pending_prefix=[])
        actions = list(pending)
        actions.append(self.mgr.execute_action(TermAction()))
        self._add_turn(actions)
        return self.turns


class AutoExplore:
    """Produce text history via proxy strategies."""

    def __init__(self, room: Room, agent: Agent, np_random: np.random.Generator | None = None, strategy: str = "oracle"):
        self.strategy = strategy
        self.proxy = AgentProxy(room, agent, strategy=strategy)

    def _format_history_to_obs(self, turns: List[Turn]) -> str:
        def ori_name(v):
            return {(0, 1): 'north', (1, 0): 'east', (0, -1): 'south', (-1, 0): 'west'}[tuple(v)]
        lines: List[str] = []
        for i, t in enumerate(turns, 1):
            lines.append(f"{i}. {action_results_to_text(t.actions)}")
        return "\n".join(lines)

    def gen_exp_history(self) -> str:
        return self._format_history_to_obs(self.proxy.run())


if __name__ == "__main__":
    from ..utils.room_utils import RoomGenerator, RoomPlotter

    room, agent = RoomGenerator.generate_room(
        room_size=[10, 10],
        n_objects=3,
        generation_type='rand',
        np_random=np.random.default_rng(42),
    )
    print(BaseAction.get_field_of_view())
    ObserveAction.MODE = 'full'
    RoomPlotter.plot(room, agent, mode='img', save_path='room.png')

    auto = AutoExplore(room, agent, strategy="oracle")
    print(auto.gen_exp_history())