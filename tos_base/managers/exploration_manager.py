import copy
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
from dataclasses import dataclass

from ..core.object import Agent
from ..actions import *
from ..core.room import Room

@dataclass
class ExplorationTurnLog:
    """Log data for a single exploration turn."""
    node_coverage: float
    edge_coverage: float
    step: int
    action_counts: Dict[str, int]
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None

    def to_dict(self):
        return {
            "node_coverage": self.node_coverage,
            "edge_coverage": self.edge_coverage,
            "step": self.step,
            "action_counts": dict(self.action_counts),
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {}
        }

class ExplorationManager:
    """Minimal exploration manager without graphs.

    - Keeps copies of `room` and `agent` for simulation.
    - Executes actions and logs turns.
    - Graph-related metrics default to safe zeros.
    """
    DEFAULT_EXP_SUMMARY = {"node_coverage": 0.0, "edge_coverage": 0.0, "n_exploration_steps": 0, "action_counts": {}}
    
    def __init__(self, room: Room, agent: Agent):
        self.base_room = room.copy()
        self.exploration_room = room.copy()
        self.agent = agent.copy()
        self.keep_object_names = [self.agent.name] + [obj.name for obj in getattr(self.exploration_room, 'all_objects', [])]

        self.finished = False
        self.exp_summary = copy.deepcopy(self.DEFAULT_EXP_SUMMARY)
        self.turn_logs: List[ExplorationTurnLog] = []
        self.history: List[ActionSequence] = []
        
        # Coverage tracking (exclude gates)
        self._init_node_name = "__init__"
        self.init_pos = self.agent.init_pos.copy()
        self._init_room_id = int(self.agent.init_room_id)

        # Node names: all objects in the exploration room
        self.node_names: List[str] = [o.name for o in self.exploration_room.objects]

        # Edge targets: per-room object pairs + (init, object-in-init-room)
        self.target_edges: Set[frozenset] = set()
        for _, names in self.exploration_room.objects_by_room.items():
            if not names:
                continue
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    self.target_edges.add(frozenset({a, b}))
        init_room_objects = self.exploration_room.objects_by_room.get(self._init_room_id, [])
        for name in init_room_objects:
            self.target_edges.add(frozenset({self._init_node_name, name}))
        
        self.observed_nodes: Set[str] = set()
        self.known_edges: Set[frozenset] = set()

        # Action counts and costs
        self.action_counts: Dict[str, int] = {}
        self.action_cost: int = 0
        # Observed names (objects and gates) to gate Move() eligibility
        self.observed_items: Set[str] = set()
        
    def _execute_and_update(self, action: BaseAction) -> ActionResult:
        """Execute action and update exploration state."""
        kwargs = {}
        # Enforce "observed-before-move"
        if isinstance(action, MoveAction):
            kwargs['observed_items'] = list(self.observed_items)
        result = action.execute(self.exploration_room, self.agent, **kwargs)
        if not result.success:
            return result
        
        # Count action, cost, and update coverage
        self.action_counts[result.action_type] = self.action_counts.get(result.action_type, 0) + 1
        if hasattr(action, 'cost'):
            try:
                self.action_cost += int(action.cost())
            except Exception:
                pass
        if isinstance(action, ObserveAction):
            self._update_coverage_from_observe(result)
        
        return result



    def execute_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result."""
        return self._execute_and_update(action)
    
    def execute_success_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result (must be successful)."""
        result = self._execute_and_update(action)
        assert result.success, f"Action {action} failed: {result.message}"
        return result

    def execute_action_sequence(self, action_sequence: ActionSequence) -> Tuple[Dict[str, Any], List[ActionResult]]:
        """
        Execute a sequence of motion actions followed by a final action.
        If any motion action fails, execute an observe action and end.
        Returns info and list of action results.
        """
        
        # Query-only sequence
        if getattr(action_sequence, 'query_actions', None):
            info, action_results = {}, []
            for action in action_sequence.query_actions:
                result = self._execute_and_update(action)
                action_results.append(result)
                info.update(result.data)
            self._log_exploration(action_sequence)
            return info, action_results

        assert action_sequence.final_action, "Action sequence requires a final action."

        info = {}
        action_results = []
        
        # Execute motion actions
        for action in action_sequence.motion_actions:
            result = self._execute_and_update(action)
            action_results.append(result)
            info.update(result.data)
            if not result.success:
                # On failure, perform an observe action and end
                obs_result = self._execute_and_update(ObserveAction())
                action_results.append(obs_result)
                assert obs_result.success, f"Observe action failed: {obs_result.message}"
                info.update(obs_result.data)
                self._log_exploration(action_sequence)
                return info, action_results

        # Execute final action
        final_action = action_sequence.final_action
        result = self._execute_and_update(final_action)
        action_results.append(result)
        assert result.success, f"Final action {final_action} failed: {result.message}"
        info.update(result.data)

        # Always log before return
        self._log_exploration(action_sequence)
        return info, action_results
    
    def finish_exploration(self, return_to_origin: bool = True) -> Room:
        """Complete exploration and return final room state."""
        if return_to_origin:
            result = self.execute_action(ReturnAction())
            if not result.success:
                raise ValueError(f"Failed to return to origin: {result.message}")
        self.finished = True
        return self.exploration_room
    
    def get_exp_summary(self) -> Dict[str, Any]:
        """Get exploration summary."""
        return dict(self._update_exp_summary())
    
    @staticmethod
    def aggregate_group_performance(exp_summaries: List[Dict]) -> Dict[str, float]:
        """Calculate exploration performance for a group."""
        if not exp_summaries:
            return {"avg_coverage": 0.0, "avg_exploration_steps": 0.0, "avg_node_coverage": 0.0, "avg_edge_coverage": 0.0}
        
        n = len(exp_summaries)
        return {
            "avg_coverage": sum(m.get('coverage', 0.0) for m in exp_summaries) / n,
            "avg_exploration_steps": sum(m.get('n_exploration_steps', 0) for m in exp_summaries) / n,
            "avg_node_coverage": sum(m.get('node_coverage', m.get('coverage', 0.0)) for m in exp_summaries) / n,
            "avg_edge_coverage": sum(m.get('edge_coverage', 0.0) for m in exp_summaries) / n,
        }
    
    # No passive history generation here; proxies produce text histories directly.
    
    # === Coverage helpers ===
    def _anchor_name(self) -> Optional[str]:
        # If standing on an object position, use that object as anchor (exclude gates)
        for obj in self.exploration_room.objects:
            if np.allclose(obj.pos, self.agent.pos):
                return obj.name
        # Initial position anchor
        if np.allclose(self.agent.pos, self.init_pos):
            return self._init_node_name
        return None

    def _update_coverage_from_observe(self, observe_result: 'ActionResult') -> None:
        visible = observe_result.data.get('visible_objects', []) or []
        # node coverage
        for name in visible:
            self.observed_items.add(name)
            if name in self.node_names:
                self.observed_nodes.add(name)
        # edge coverage: observe A from B (B is anchor)
        anchor = self._anchor_name()
        if not anchor:
            return
        for name in visible:
            if name == anchor:
                continue
            pair = frozenset({anchor, name})
            if pair in self.target_edges:
                self.known_edges.add(pair)

    def _update_coverage_from_query(self, query_result: 'ActionResult') -> None:
        # Coverage: two nodes + edge between them
        objs = query_result.data.get('objects') or query_result.data.get('pair') or []
        if len(objs) == 2:
            a, b = objs[0], objs[1]
            if a in self.node_names:
                self.observed_nodes.add(a)
            if b in self.node_names:
                self.observed_nodes.add(b)
            pair = frozenset({a, b})
            if pair in self.target_edges:
                self.known_edges.add(pair)


    
    def _log_exploration(self, action_sequence: ActionSequence) -> None:
        """Log exploration history and efficiency."""
        # Log current turn with coverage snapshot
        self._update_exp_summary()
        step_idx = len(self.turn_logs) + 1
        turn_log = ExplorationTurnLog(
            node_coverage=self.exp_summary.get('node_coverage', 0.0),
            edge_coverage=self.exp_summary.get('edge_coverage', 0.0),
            step=step_idx,
            action_counts=dict(self.exp_summary.get('action_counts', {})),
            room_state=self.exploration_room.copy(),
            agent_state=self.agent.copy()
        )
        self.turn_logs.append(turn_log)
        self.history.append(action_sequence)
    
    def _update_exp_summary(self) -> Dict[str, Any]:
        """Calculate current coverage and summary stats."""
        n_nodes = len(self.node_names) or 1
        node_cov = len(self.observed_nodes) / n_nodes
        edge_den = len(self.target_edges) or 1
        edge_cov = len(self.known_edges) / edge_den
        self.exp_summary = {
            "node_coverage": node_cov,
            "edge_coverage": edge_cov,
            "n_exploration_steps": len(self.turn_logs),
            "action_counts": dict(self.action_counts),
            "action_cost": int(self.action_cost),
        }
        return self.exp_summary
    

if __name__ == "__main__":
    pass