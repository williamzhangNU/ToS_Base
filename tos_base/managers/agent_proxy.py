import copy
from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np

from ..core.room import Room
from ..actions.base import BaseAction
from ..actions.actions import MoveAction, RotateAction, ObserveAction, TermAction, ObserveApproxAction, QueryAction, ReturnAction
from .spatial_solver import SpatialSolver
from ..core.relationship import CardinalBinsAllo
from .exploration_manager import ExplorationManager
from ..core.object import Agent
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
    """Base proxy that executes actions via ExplorationManager and logs a simple history."""

    def __init__(self, room: Room, agent: Agent):
        self.mgr = ExplorationManager(room, agent)
        self.room, self.agent = self.mgr.exploration_room, self.mgr.agent
        # Focused room id for planning when at gates (agent may belong to two rooms at a door)
        self.room_focus = self.agent.room_id

        self.gates_by_room: Dict[int, Set[str]] = {int(r): set(glist) for r, glist in self.room.gates_by_room.items()}

        # nodes with and without gates
        self.object_nodes_by_room: Dict[int, Set[str]] = {}
        for o in self.room.objects:
            self.object_nodes_by_room.setdefault(int(o.room_id), set()).add(o.name)
        # Include rooms that have only gates (no objects)
        room_ids = set(self.object_nodes_by_room.keys()) | {int(r) for r in self.gates_by_room.keys()}
        self.nodes_by_room: Dict[int, Set[str]] = {
            int(rid): set(self.object_nodes_by_room.get(int(rid), set())) | self.gates_by_room.get(int(rid), set())
            for rid in room_ids
        }
        self.known_nodes_by_room: Dict[int, Set[str]] = {int(rid): set() for rid in self.nodes_by_room}

        self.all_edges_by_room: Dict[int, Set[frozenset]] = {}
        for rid, names in self.nodes_by_room.items():
            pairs: Set[frozenset] = set()
            for a in names:
                for b in names:
                    if a != b:
                        pairs.add(frozenset({a, b}))
            self.all_edges_by_room[int(rid)] = pairs
        self.known_edges_by_room: Dict[int, Set[frozenset]] = {int(rid): set() for rid in self.all_edges_by_room}

        # add edges from initial position to objects in the starting room
        self.initial_anchor: str = "__start__"
        start_rid = self._current_room()
        for obj_name in self.nodes_by_room.get(start_rid, set()):
            self.all_edges_by_room[start_rid].add(frozenset({self.initial_anchor, obj_name}))

        self.total_object_nodes: int = sum(len(v) for v in self.object_nodes_by_room.values())

        self.turns: List[Turn] = []
        self.visited: Set[int] = set()
        self.current_gate: str | None = None
        self.anchor: str | None = None

    # --- helpers ---
    def _current_room(self) -> int:
        return int(self.room_focus)

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
        vis = last_result.data.get("visible_objects", [])
        for n in vis:
            self.known_nodes_by_room.setdefault(rid, set()).add(n)
        # Also record edges if an anchor (gate or node) is set
        if self.anchor:
            known = self.known_edges_by_room.setdefault(rid, set())
            for n in vis:
                if n != self.anchor:
                    known.add(frozenset({self.anchor, n}))

    def _rotate_by(self, delta: int) -> List:
        assert delta is not None, "Invalid rotation delta"
        delta = int(delta) % 360
        assert delta % 90 == 0, "Invalid rotation delta"
        if delta > 180:
            delta -= 360
        return [] if delta == 0 else [self.mgr.execute_success_action(RotateAction(delta))]


    def _rotate_to_face(self, target_pos: np.ndarray) -> List:
        vec = target_pos - self.agent.pos
        if np.allclose(vec, 0):
            return []
        desired = _closest_cardinal(vec)
        cur, des = _ori_to_deg(self.agent.ori), _ori_to_deg(desired)
        return self._rotate_by(des - cur)

    def _rotate_to_ori(self, desired_ori: np.ndarray) -> List:
        cur, des = _ori_to_deg(self.agent.ori), _ori_to_deg(desired_ori)
        return self._rotate_by(des - cur)

    def _move_to(self, name: str) -> List:
        target = self.room.get_object_by_name(name)
        acts = []
        acts += self._rotate_to_face(target.pos)
        acts.append(self.mgr.execute_success_action(MoveAction(name)))
        return acts

    def _observe(self, prefix_actions: List = None) -> None:
        acts = list(prefix_actions or [])
        obs = self.mgr.execute_success_action(ObserveApproxAction())
        acts.append(obs)
        self._update_known_from_observe(obs)
        self._add_turn(acts)

    def _unknown_nodes_in_room(self, rid: int) -> Set[str]:
        return self.nodes_by_room.get(rid, set()) - self.known_nodes_by_room.get(rid, set())

    def _unknown_objects_in_room(self, rid: int) -> Set[str]:
        return self.object_nodes_by_room.get(rid, set()) - self.known_nodes_by_room.get(rid, set())

    def _unknown_edges_in_room(self, rid: int) -> Set[frozenset]:
        return self.all_edges_by_room.get(rid, set()) - self.known_edges_by_room.get(rid, set())

    def _score_rotation_nodes(self, rot: int, unknown_nodes: Set[str]) -> int:
        """How many unknown nodes become visible after rot."""
        if not unknown_nodes:
            return 0
        objects = {o.name: o for o in self.room.all_objects}
        R = BaseAction._get_rotation_matrix(rot)
        tmp = self.agent.copy(); tmp.ori = self.agent.ori @ R
        return sum(1 for n in unknown_nodes if n in objects and BaseAction._is_visible(tmp, objects[n]))

    def _score_rotation_edges(self, rot: int, anchor: str, unknown_edges: Set[frozenset]) -> int:
        """How many unknown edges incident to anchor become visible after rot."""
        anchored = {e for e in unknown_edges if anchor in e}
        if not anchored:
            return 0
        objects = {o.name: o for o in self.room.all_objects}
        R = BaseAction._get_rotation_matrix(rot)
        tmp = self.agent.copy(); tmp.ori = self.agent.ori @ R
        visible = {n for n, o in objects.items() if BaseAction._is_visible(tmp, o)}
        return sum(1 for e in anchored if next(iter(e - {anchor})) in visible)

    def _score_rotation(self, rot: int) -> int:
        """Unified rotation score: edges if anchor set, otherwise nodes."""
        rid = self._current_room()
        if self.anchor:
            return self._score_rotation_edges(rot, self.anchor, self._unknown_edges_in_room(rid))
        return self._score_rotation_nodes(rot, self._unknown_nodes_in_room(rid))

    def _gate_between(self, a: int, b: int) -> str:
        for gate_name, rooms in self.room.rooms_by_gate.items():
            if set(rooms) == {int(a), int(b)}:
                return gate_name
        raise AssertionError(f"No gate between rooms {a} and {b}")

    def _traverse_to(self, next_rid: int) -> List:
        # Move to the connecting gate and face next room; at gate, agent can see both rooms.
        cur = self._current_room()
        gate_name = self._gate_between(cur, next_rid)
        gate_obj = self.room.get_object_by_name(gate_name)
        acts = []
        if not np.allclose(self.agent.pos, gate_obj.pos):
            acts += self._rotate_to_face(gate_obj.pos)
            acts.append(self.mgr.execute_success_action(MoveAction(gate_name)))
        # face next room; TODO may not be needed
        acts += self._rotate_to_ori(-gate_obj.get_ori_for_room(int(cur)))
        self.current_gate = gate_name
        self.room_focus = int(next_rid)
        self.known_nodes_by_room.setdefault(next_rid, set()).add(gate_name)
        return acts

    def _allowed_rotations(self, is_initial: bool, continuous_rotation: bool = True) -> tuple:
        fov = BaseAction.get_field_of_view()
        if continuous_rotation:
            # note continuous rotation: 0 -> (+90) -> 90 -> (+90) -> 180 -> (+90) -> 270
            return (0, 90, 90, 90) if is_initial else ((0,) if fov == 180 else (0, 90, 180))
        else:
            # not continuous: 0, 0 -> 90, 0 -> 180, 0 -> 270
            return (0, 90, 180, 270) if is_initial else ((0,) if fov == 180 else (0, 90, 270))

    def _all_objects_known_globally(self) -> bool:
        # early stop when all objects (exclude gates) are known
        known_total = 0
        for rid, objs in self.object_nodes_by_room.items():
            known_total += len(objs & self.known_nodes_by_room.get(int(rid), set()))
        return known_total >= self.total_object_nodes

    def _subtree_has_nodes(self, start_rid: int) -> bool:
        """check if the subtree has any nodes. Start from start_rid"""
        stack = [int(start_rid)]
        seen = set(copy.deepcopy(self.visited))
        while stack:
            r_id = int(stack.pop())
            if r_id in seen:
                continue
            seen.add(r_id)
            if len(self.object_nodes_by_room.get(r_id, set())) > 0:
                return True
            stack.extend(int(adj_r_id) for adj_r_id in self.room.adjacent_rooms_by_room.get(r_id, []))
        return False

    # hooks for subclasses
    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def _explore_room(self, is_initial: bool, prefix_actions: List = None) -> None:
        # default: entry observes only
        self._on_entry_observe(is_initial=is_initial, prefix_actions=prefix_actions)
    
    def _prune_dfs(self, rid: int): # if prune subtree from rid
        return False

    def _dfs(self, rid: int, is_initial: bool, pre_actions: List = None) -> List:
        """DFS over rooms.
        - pre_actions: actions before first observe in this room
        - carry_actions: if no observe here, carry pre_actions into child
        - visited: expanded rooms; stop when all objects are known
        """
        self.visited.add(rid)
        before = len(self.turns)
        self._explore_room(is_initial=is_initial, prefix_actions=pre_actions or [])
        observed_here = len(self.turns) > before
        carry_actions: List = [] if observed_here else list(pre_actions or [])
        if self._all_objects_known_globally():
            assert not carry_actions, "Carry must be empty if all objects are known"
            return carry_actions
        for child_rid in sorted(self.room.adjacent_rooms_by_room.get(rid, [])):
            if child_rid in self.visited:
                continue
            if self._prune_dfs(int(child_rid)):
                continue
            to_child = carry_actions + self._traverse_to(int(child_rid))
            carry_actions = self._dfs(int(child_rid), is_initial=False, pre_actions=to_child)
            # Stop here if exploration is complete; avoid returning to parent.
            if self._all_objects_known_globally():
                return carry_actions
            carry_actions = carry_actions + self._traverse_to(int(rid))
        return carry_actions

    def run(self) -> List[Turn]:  # to be used by subclasses too
        start_rid = self._current_room()
        _ = self._dfs(start_rid, is_initial=True, pre_actions=[])
        # final turn contains only Term()
        self._add_turn([self.mgr.execute_success_action(TermAction())])
        return self.turns

    def to_text(self, image_placeholder = None) -> str:
        lines: List[str] = []
        for i, t in enumerate(self.turns, 1):
            lines.append(f"{i}. {action_results_to_text(t.actions, image_placeholder)}")
        return "\n".join(lines)
    
class OracleAgentProxy(AgentProxy):
    """Oracle Agent: greedy rotations to reveal all nodes (knows everything about nodes)."""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        rid = self._current_room()
        allowed = self._allowed_rotations(is_initial, continuous_rotation=False)
        while self._unknown_nodes_in_room(rid):
            targets = self._unknown_nodes_in_room(rid)
            scores = {d: self._score_rotation_nodes(d, targets) for d in allowed}
            best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
            
            if best_score == 0:
                break
            self._observe((prefix_actions or []) + self._rotate_by(best))
            prefix_actions = []

    def _prune_dfs(self, rid: int):
        return not self._subtree_has_nodes(rid)

# TODO fix unknown objects in room

class StrategistAgentProxy(AgentProxy):
    """NodeSweeper: simple sweep rotations; may not be optimal or complete."""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        for d in self._allowed_rotations(is_initial):
            if self._all_objects_known_globally():
                break
            self._observe((prefix_actions or []) + self._rotate_by(d))
            prefix_actions = []


class InquisitorAgentProxy(AgentProxy):
    """Inquisitor Agent: visit/confirm all edges between nodes (know nothing in prior)"""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        # treat entry gate as anchor; at start, use initial anchor
        self.anchor = self.initial_anchor if is_initial else self.current_gate
        for d in self._allowed_rotations(is_initial):
            if self._all_objects_known_globally(): # early stop if see all objects at entry
                break
            self._observe((prefix_actions or []) + self._rotate_by(d))
            prefix_actions = []
        self.anchor = None

    # move to each node with unknown edges and observe
    def _resolve_edges(self) -> None:
        rid = self._current_room()
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return
        objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)
        while objects_with_unknown_edges:
            anchor = objects_with_unknown_edges.pop()
            self._observe(self._move_to(anchor))
            for d in (0, 90, 90, 90):
                if d == 0:
                    continue
                self._observe(self._rotate_by(d))
                if not any(anchor in p for p in self._unknown_edges_in_room(rid)): # no unknown edges from anchor
                    break
            objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)

    # how to choose anchor node with unknown edges
    def _anchors_with_unknown_edges(self, rid: int) -> List[str]:
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return []
        anchors = sorted(self.nodes_by_room.get(rid, set()))
        return [a for a in anchors if any(a in p for p in unknown)]

    def _explore_room(self, is_initial: bool, prefix_actions: List = None) -> None:
        self._on_entry_observe(is_initial=is_initial, prefix_actions=prefix_actions)
        self._resolve_edges()

class GreedyInquisitorAgentProxy(InquisitorAgentProxy):
    """Greedy Inquisitor Agent: pick best rotation each step."""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        self.anchor = self.initial_anchor if is_initial else self.current_gate
        rid = self._current_room()
        while any(self.anchor in p for p in self._unknown_edges_in_room(rid)):
            scores = {d: self._score_rotation(d) for d in (0, 90, 180, 270)}
            best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
            if best_score == 0:
                break
            self._observe((prefix_actions or []) + self._rotate_by(best))
            prefix_actions = []
        self.anchor = None

    def _resolve_edges(self) -> None:
        # move to each node with unknown edges; at each node, rotate greedily by edge score
        rid = self._current_room()
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return
        objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)
        while objects_with_unknown_edges:
            anchor = objects_with_unknown_edges.pop()
            self.anchor = anchor
            prefix_actions = self._move_to(anchor)
            while any(anchor in p for p in self._unknown_edges_in_room(rid)): # exist unknown edges from anchor
                scores = {d: self._score_rotation(d) for d in (0, 90, 180, 270)}
                best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
                if best_score == 0:
                    break
                self._observe(prefix_actions + self._rotate_by(best))
                prefix_actions = []
            objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)
            self.anchor = None


class AnalystAgentProxy(AgentProxy):
    """Analyst Agent: observe, then greedily query to reduce discrete (cardinal-bin) relationships."""

    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None,
                 max_queries: int = 16, rel_threshold: int = 0, eval_samples: int = 30,
                 delegate: str = 'oracle'):
        super().__init__(room, agent)
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.solver = SpatialSolver([o.name for o in self.room.all_objects] + ['initial_pos'], grid_size=(g if grid_size is None else grid_size))
        self.max_queries = int(max_queries)
        self.rel_threshold = int(rel_threshold)
        self.eval_samples = int(eval_samples)
        self.delegate = (delegate or 'oracle').lower()

    def _ingest_observations(self) -> None:
        for i, t in enumerate(self.turns):
            obs = t.actions[-1]
            triples = obs.data.get('relation_triples', []) if hasattr(obs, 'data') else []
            if triples:
                self.solver.add_observation(triples)

    def _query_object(self, obj: str) -> None:
        res = self.mgr.execute_success_action(QueryAction(obj))
        self._add_turn([res])
        triples = res.data.get('relation_triples', []) if hasattr(res, 'data') else []
        self.solver.add_observation(triples)

    def _current_metrics(self) -> tuple[Dict[str, int], int, Dict[tuple, Set[str]], int]:
        """Current discrete metrics using cardinal bins."""
        return self.solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())

    # ---- Greedy global query with simulation ----
    def _simulate_query_gain(self, obj: str, baseline_rels: int) -> float:
        """Simulate query on a solver copy; return reduction in relationship count."""
        if obj != 'initial_pos' and (not self.room.has_object(obj)):
            raise ValueError(f"Object {obj} not found in room")
        sim_solver = self.solver.copy()
        res = QueryAction(obj).execute(self.room, self.agent)
        triples = res.data.get('relation_triples', [])
        if triples:
            sim_solver.add_observation(triples)
        _, _, _, new_rels = sim_solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())
        return (baseline_rels - new_rels)

    def _global_query_loop(self) -> None:
        """Greedy selection by max relationship reduction."""
        q = 0
        while q < self.max_queries:
            _, _, rel_sets, total_rels = self._current_metrics()
            if total_rels <= self.rel_threshold:
                break
            best_obj, best_gain = self._best_query(rel_sets, total_rels)
            if best_gain <= 1e-9 or best_obj is None:
                break
            self._query_object(best_obj)
            q += 1

    def _best_query(self, rel_sets: dict, total_rels: int) -> tuple:
        """Pick object with highest simulated relationship gain."""
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        scores = []
        for obj in names:
            best_rel = 0
            for other in names:
                if other == obj:
                    continue
                key = (obj, other) if (obj, other) in rel_sets else (other, obj)
                best_rel = max(best_rel, len(rel_sets.get(key, set())))
            scores.append((obj, best_rel))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_objs = [obj for obj, _ in scores[:20]]

        best_obj, best_gain = None, -1.0
        for obj in top_objs:
            gain = self._simulate_query_gain(obj, total_rels)
            if gain > best_gain:
                best_gain, best_obj = gain, obj
        return best_obj, best_gain

    def run(self) -> List[Turn]:
        """Run analyst: observe with oracle, then query."""
        # Overview observe via selected delegate
        mapping = {
            'oracle': OracleAgentProxy,
            'strategist': StrategistAgentProxy,
            'inquisitor': InquisitorAgentProxy,
            'greedy_inquisitor': GreedyInquisitorAgentProxy,
            'greedy': GreedyInquisitorAgentProxy,
        }
        DelegateCls = mapping.get(self.delegate, OracleAgentProxy)
        delegate = DelegateCls(self.room, self.agent)
        d_turns = delegate.run()
        self.mgr, self.room, self.agent = delegate.mgr, delegate.mgr.exploration_room, delegate.mgr.agent
        self.turns = list(d_turns[:-1]) if d_turns else [] # drop final Term()

        # Ingest all observed relation triples into solver
        self._ingest_observations()
        # Ensure return to initial state before queries
        ret = self.mgr.execute_success_action(ReturnAction())
        self._add_turn([ret])
        # Global greedy queries
        self._global_query_loop()
        # Terminate
        self._add_turn([self.mgr.execute_success_action(TermAction())])
        return self.turns

def get_agent_proxy(name: str, room: Room, agent: Agent, delegate: str | None = None) -> AgentProxy:
    name = (name or 'oracle').lower()
    mapping = {
        'oracle': OracleAgentProxy,
        'strategist': StrategistAgentProxy,
        'inquisitor': InquisitorAgentProxy,
        'greedy_inquisitor': GreedyInquisitorAgentProxy,
        'analyst': AnalystAgentProxy,
    }
    if name == 'analyst':
        return mapping['analyst'](room, agent, delegate=(delegate or 'oracle'))
    return mapping.get(name, OracleAgentProxy)(room, agent)


if __name__ == "__main__":
    from ..utils.room_utils import RoomGenerator, RoomPlotter
    from tqdm import tqdm
    for seed in tqdm(range(9, 10)):
        room, agent = RoomGenerator.generate_room(
            room_size=[10, 10],
            n_objects=3,
            np_random=np.random.default_rng(seed),
            level=0,
            main=6
        )
        print(room)
        print(room.mask)
        print(room.gates)
        print(agent)
        ObserveAction.MODE = 'full'
        RoomPlotter.plot(room, agent, mode='img', save_path='room.png')

        # proxy = OracleAgentProxy(room, agent)        # node_seeker (complete nodes)
        # proxy = StrategistAgentProxy(room, agent)    # node_sweeper
        # proxy = InquisitorAgentProxy(room, agent)    # edge_seeker (complete edges)
        # proxy = GreedyInquisitorAgentProxy(room, agent) # greedy_edge_seeker
        proxy = AnalystAgentProxy(room, agent, delegate='greedy_inquisitor')
        proxy.run()
        print(proxy.to_text())
        print(proxy.mgr.get_exp_summary())