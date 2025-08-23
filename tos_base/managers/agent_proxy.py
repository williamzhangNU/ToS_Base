import copy
from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np

from ..core.room import Room
from ..actions.base import BaseAction
from ..actions.actions import MoveAction, RotateAction, ObserveAction, TermAction, GoThroughDoorAction, ObserveApproxAction, QueryRelAction
from .spatial_solver import SpatialSolver, Variable, Constraint, AC3Solver
from ..core.relationship import PairwiseRelationship, RelationTriple
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
        return [] if delta == 0 else [self.mgr.execute_action(RotateAction(delta))]


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
        acts.append(self.mgr.execute_action(MoveAction(name)))
        return acts

    def _observe(self, prefix_actions: List = None) -> None:
        acts = list(prefix_actions or [])
        # obs = self.mgr.execute_action(ObserveAction())
        obs = self.mgr.execute_action(ObserveApproxAction())
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
        cur = self._current_room()
        gate_name = self._gate_between(cur, next_rid)
        gate_obj = self.room.get_object_by_name(gate_name)
        acts = []
        # If already at the gate, skip moving and rotating
        if not np.allclose(self.agent.pos, gate_obj.pos):
            acts += self._rotate_to_face(gate_obj.pos)
            acts.append(self.mgr.execute_action(MoveAction(gate_name)))
        # Must face doorway: opposite of gate ori for current room (acceptable alternative is same as other room ori)
        acts += self._rotate_to_ori(-gate_obj.get_ori_for_room(int(cur)))
        go = self.mgr.execute_action(GoThroughDoorAction(gate_name))
        acts.append(go)
        self.current_gate = gate_name
        self.known_nodes_by_room.setdefault(next_rid, set()).add(gate_name) # next room gate is known in next room
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
        self._add_turn([self.mgr.execute_action(TermAction())])
        return self.turns

    def to_text(self) -> str:
        lines: List[str] = []
        for i, t in enumerate(self.turns, 1):
            lines.append(f"{i}. {action_results_to_text(t.actions)}")
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
        for anchor in self._anchors_with_unknown_edges(rid):
            self._observe(self._move_to(anchor))
            for d in (0, 90, 90, 90):
                if d == 0:
                    continue
                self._observe(self._rotate_by(d))
                if not any(anchor in p for p in self._unknown_edges_in_room(rid)): # no unknown edges from anchor
                    break

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
        for anchor in self._anchors_with_unknown_edges(rid):
            self.anchor = anchor
            prefix_actions = self._move_to(anchor)
            while any(anchor in p for p in self._unknown_edges_in_room(rid)): # exist unknown edges from anchor
                scores = {d: self._score_rotation(d) for d in (0, 90, 180, 270)}
                best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
                if best_score == 0:
                    break
                self._observe(prefix_actions + self._rotate_by(best))
                prefix_actions = []

            self.anchor = None


class AnalystAgentProxy(AgentProxy):
    """Analyst Agent: observe (approx) to see nodes, then greedily query allocentric pairs.
    Greedy criterion switches between reducing relationships vs positions.
    """

    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None,
                 max_queries: int = 16, rel_threshold: int = 0, pos_threshold: int | None = None,
                 max_eval_pairs: int = 20, eval_samples: int = 30, observe_strategy: str = 'oracle',
                 metric_type: str = 'allocentric_bins'):
        super().__init__(room, agent)
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.solver = SpatialSolver([o.name for o in self.room.all_objects] + ['initial_pos'], grid_size=(g if grid_size is None else grid_size))
        self.metric_mode = 'relationships'
        self.metric_type = metric_type  # 'relationships', 'allocentric_bins'
        self.max_queries = int(max_queries)
        self.rel_threshold = int(rel_threshold)
        self.pos_threshold = int(pos_threshold) if pos_threshold is not None else len(self.room.all_objects)
        self.max_eval_pairs = int(max_eval_pairs)
        self.eval_samples = int(eval_samples)
        self.observe_strategy = observe_strategy  # 'oracle' or 'strategist'
        self.phase = 'observe'

        self.solver.set_initial_position('initial_pos', (0, 0)) # agent initial position unknown, so set to (0, 0)

    def _ingest_from_turns(self) -> None:
        for i, t in enumerate(self.turns):
            obs = t.actions[-1]
            triples = obs.data.get('relation_triples', []) if hasattr(obs, 'data') else []
            if triples:
                self.solver.add_observation(triples)

    def _query_and_ingest(self, a: str, b: str) -> None:
        res = self.mgr.execute_action(QueryRelAction(a, b))
        self._add_turn([res])
        o1_pos = self.room.get_object_by_name(a).pos if a != 'initial_pos' else self.agent.init_pos
        o2_pos = self.room.get_object_by_name(b).pos if b != 'initial_pos' else self.agent.init_pos
        rel = PairwiseRelationship.relationship(tuple(o1_pos), tuple(o2_pos), anchor_ori=tuple(self.agent.init_ori), full=True)
        self.solver.add_observation([RelationTriple(subject=a, anchor=b, relation=rel, orientation=tuple(self.agent.init_ori))])

    def _compute_metrics(self) -> tuple[Dict[str, int], int, Dict[tuple, Set[str]], int]:
        """Compute current solver metrics for query selection."""
        if self.metric_type == 'allocentric_bins':
            return self.solver.compute_allocentric_bin_metrics(max_samples_per_var=self.eval_samples)
        else:
            return self.solver.compute_metrics('dir', max_samples_per_var=self.eval_samples)

    # ---- Greedy global query with simulation ----
    def _simulate_effect(self, a: str, b: str, mode: str,
                         baseline_rels: int, baseline_pos: int) -> tuple[float, float]:
        """Efficiently simulate adding a constraint using incremental solver."""
        # Get relationship between objects
        o1_pos = self.room.get_object_by_name(a).pos if a != 'initial_pos' else self.agent.init_pos
        o2_pos = self.room.get_object_by_name(b).pos if b != 'initial_pos' else self.agent.init_pos
        north_ori = tuple(self.agent.init_ori)
        rel = PairwiseRelationship.relationship(tuple(o1_pos), tuple(o2_pos), anchor_ori=north_ori, full=True)
        
        # Use efficient copy and incremental constraint addition
        sim_solver = self.solver.copy()
        sim_solver.add_observation([RelationTriple(subject=a, anchor=b, relation=rel, orientation=north_ori)])
        
        # Compute new metrics
        if self.metric_type == 'allocentric_bins':
            _, new_pos, _, new_rels = sim_solver.compute_allocentric_bin_metrics(max_samples_per_var=self.eval_samples)
        else:
            _, new_pos, _, new_rels = sim_solver.compute_metrics('dir', max_samples_per_var=self.eval_samples)
        
        return (baseline_rels - new_rels), (baseline_pos - new_pos)

    def _global_query_loop(self) -> None:
        """Optimized greedy query selection loop using incremental metrics."""
        q = 0
        switched = False
        
        while q < self.max_queries:
            # Compute current metrics using dedicated function
            dom_sizes, total_pos, rel_sets, total_rels = self._compute_metrics()
            
            # Check termination criteria
            if total_rels <= self.rel_threshold and total_pos <= self.pos_threshold:
                break
                
            # Get candidate pairs and score them efficiently
            best_pair, best_gain = self._find_best_query_pair(dom_sizes, rel_sets, total_rels, total_pos)
            
            # Switch mode if no good queries found, TODO update here for stop criteria
            if best_gain <= 1e-9:
                if switched:
                    break
                self.metric_mode = 'positions' if self.metric_mode == 'relationships' else 'relationships'
                switched = True
                continue
                
            # Execute best query
            self._query_and_ingest(best_pair[0], best_pair[1])
            q += 1

    def _find_best_query_pair(self, dom_sizes: dict, rel_sets: dict, total_rels: int, total_pos: int) -> tuple:
        """Find the best query pair by efficient candidate ranking and simulation."""
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        
        # Score and sort candidates
        candidates = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if self.metric_mode == 'positions':
                    score = dom_sizes.get(a, 0) * dom_sizes.get(b, 0)
                else:
                    key = (a, b) if (a, b) in rel_sets else (b, a)
                    score = len(rel_sets.get(key, set()))
                candidates.append(((a, b), score))
        
        # Sort by score and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [pair for pair, _ in candidates[:self.max_eval_pairs]]
        
        # Simulate each candidate and find best
        best_pair, best_gain = (None, None), -1.0
        for a, b in top_candidates:
            d_rel, d_pos = self._simulate_effect(a, b, self.metric_mode, total_rels, total_pos)
            gain = d_rel if self.metric_mode == 'relationships' else d_pos
            if gain > best_gain:
                best_gain, best_pair = gain, (a, b)
                
        return best_pair, best_gain

    def run(self) -> List[Turn]:
        """Run analyst: delegate exploration to oracle/strategist, then query."""
        # Delegate full exploration (_dfs) to ensure identical behavior
        delegate_cls = OracleAgentProxy if self.observe_strategy == 'oracle' else StrategistAgentProxy
        delegate = delegate_cls(self.room, self.agent)
        d_turns = delegate.run()
        # adopt delegate manager/state for accurate history/costs; drop its final Term()
        self.mgr, self.room, self.agent = delegate.mgr, delegate.mgr.exploration_room, delegate.mgr.agent
        self.turns = list(d_turns[:-1]) if d_turns else []
        # Ingest all observed relation triples into solver
        self._ingest_from_turns()
        # Global greedy queries
        self._global_query_loop()
        # Terminate
        self._add_turn([self.mgr.execute_action(TermAction())])
        return self.turns

def get_agent_proxy(name: str, room: Room, agent: Agent) -> AgentProxy:
    name = (name or 'oracle').lower()
    mapping = {
        'oracle': OracleAgentProxy,
        'strategist': StrategistAgentProxy,
        'inquisitor': InquisitorAgentProxy,
        'greedy_inquisitor': GreedyInquisitorAgentProxy,
        'analyst': AnalystAgentProxy,
    }
    return mapping.get(name, OracleAgentProxy)(room, agent)


if __name__ == "__main__":
    from ..utils.room_utils import RoomGenerator, RoomPlotter
    from tqdm import tqdm
    for seed in tqdm(range(1)):
        room, agent = RoomGenerator.generate_room(
            room_size=[12, 12],
            n_objects=5,
            np_random=np.random.default_rng(seed),
            level=1,
            main=5
        )
        print(room)
        print(room.gates)
        ObserveAction.MODE = 'full'
        RoomPlotter.plot(room, agent, mode='img', save_path='room.png')

        # proxy = OracleAgentProxy(room, agent)        # node_seeker (complete nodes)
        # proxy = StrategistAgentProxy(room, agent)    # node_sweeper
        # proxy = InquisitorAgentProxy(room, agent)    # edge_seeker (complete edges)
        # proxy = GreedyInquisitorAgentProxy(room, agent) # greedy_edge_seeker
        proxy = AnalystAgentProxy(room, agent)
        proxy.run()
        print(proxy.to_text())
        print(proxy.mgr.get_exp_summary())