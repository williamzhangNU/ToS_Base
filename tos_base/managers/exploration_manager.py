import copy
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..core.object import Agent
from ..actions import *
from ..core.room import Room

@dataclass
class ExplorationTurnLog:
    """Log data for a single exploration turn."""
    coverage: float
    redundancy: float
    n_valid_queries: int
    n_redundant_queries: int
    is_redundant: bool
    action_info: Dict[str, Any]
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None

    def to_dict(self):
        return {
            "is_redundant": self.is_redundant,
            "coverage": self.coverage,
            "redundancy": self.redundancy,
            "n_valid_queries": self.n_valid_queries,
            "n_redundant_queries": self.n_redundant_queries,
            "action_info": self.action_info,
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {}
        }

class ExplorationManager:
    """Minimal exploration manager without graphs.

    - Keeps copies of `room` and `agent` for simulation.
    - Executes actions and logs turns.
    - Graph-related metrics default to safe zeros.
    """
    DEFAULT_EXP_SUMMARY = {
        "coverage": 0, # coverage of the exploration
        "redundancy": 0, # redundancy of the exploration
        "n_valid_queries": 0, # number of valid queries
        "n_redundant_queries": 0, # number of redundant queries
    }
    
    def __init__(self, room: Room, agent: Agent):
        self.base_room = room.copy()
        self.exploration_room = room.copy()
        self.agent = agent.copy()
        self.keep_object_names = [self.agent.name] + [obj.name for obj in getattr(self.exploration_room, 'all_objects', [])]

        self.finished = False
        self.exp_summary = copy.deepcopy(self.DEFAULT_EXP_SUMMARY)
        self.turn_logs: List[ExplorationTurnLog] = []
        self.history: List[ActionSequence] = []
        
    # Graph utilities removed in simplified manager
    
    def _update_move(self, target_name: str):
        """No-op in graph-free implementation."""
        return None

    def _update_rotate(self, degrees: int):
        """No-op in graph-free implementation."""
        return None
    
    def _update_observe(self) -> bool:
        """Graph-free observe update: return non-redundant by default."""
        return False

    def _execute_and_update(self, action: BaseAction) -> ActionResult:
        """Execute action and update exploration state."""
        kwargs = {}
        result = action.execute(self.exploration_room, self.agent, **kwargs)
        
        if not result.success:
            return result
        
        # success execution
        if isinstance(action, MoveAction):
            self._update_move(result.data.get('target_name', ''))
        elif isinstance(action, RotateAction):
            self._update_rotate(result.data.get('degrees', 0))
        elif isinstance(action, ReturnAction):
            self._update_move(result.data.get('target_name', ''))
            self._update_rotate(result.data.get('degrees', 0))
        elif isinstance(action, ObserveAction):
            result.data['redundant'] = self._update_observe()
        
        return result



    def execute_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result."""
        return self._execute_and_update(action)

    def execute_action_sequence(self, action_sequence: ActionSequence) -> Tuple[Dict[str, Any], List[ActionResult]]:
        """
        Execute a sequence of motion actions followed by a final action.
        If any motion action fails, execute an observe action and end.
        Returns info and list of action results.
        """
        
        assert action_sequence.final_action, "Action sequence requires a final action."

        info = {'redundant': False}
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
                self._log_exploration(action_sequence, info)
                return info, action_results

        # Execute final action
        final_action = action_sequence.final_action
        result = self._execute_and_update(final_action)
        action_results.append(result)
        assert result.success, f"Final action {final_action} failed: {result.message}"
        info.update(result.data)

        # Always log before return
        self._log_exploration(action_sequence, info)
        return info, action_results
    
    def finish_exploration(self, return_to_origin: bool = True, keep_object_names: List[str] | None = None) -> Room:
        """Complete exploration and return final room state."""
        if return_to_origin:
            result = self.execute_action(ReturnAction())
            if not result.success:
                raise ValueError(f"Failed to return to origin: {result.message}")
        self.finished = True
        return self.exploration_room
    
    def get_exp_summary(self) -> Dict[str, Any]:
        """Get exploration summary."""
        return {**self._update_exp_summary(), "n_exploration_steps": len(self.turn_logs)}
    
    @staticmethod
    def aggregate_group_performance(exp_summaries: List[Dict]) -> Dict[str, float]:
        """Calculate exploration performance for a group."""
        if not exp_summaries:
            return {"avg_coverage": 0.0, "avg_redundancy": 0.0, "avg_exploration_steps": 0.0}
        
        return {
            "avg_coverage": sum(m.get('coverage', 0) for m in exp_summaries) / len(exp_summaries),
            "avg_redundancy": sum(m.get('redundancy', 0) for m in exp_summaries) / len(exp_summaries),
            "avg_exploration_steps": sum(m.get('n_exploration_steps', 0) for m in exp_summaries) / len(exp_summaries)
        }
    
    def get_unknown_pairs(self, keep_object_names: List[str] | None = None) -> List[Tuple[str, str]]:
        """Graph-free: no relationship tracking; return empty list."""
        return []
    
    def get_inferable_pairs(self, keep_object_names: List[str] | None = None) -> List[Tuple[str, str]]:
        """Graph-free: no inference; return empty list."""
        return []
    
    def generate_passive_history(self, strategy: str = "oracle") -> str:
        """Generate exploration history text using a proxy strategy.
        Used when env is in passive mode.
        """
        from .agent_proxy import AutoExplore
        return AutoExplore(self.base_room.copy(), self.agent.copy(), None, strategy=strategy).gen_exp_history()
    

    
    def _log_exploration(self, action_sequence: ActionSequence, info: Dict[str, Any]) -> None:
        """Log exploration history and efficiency."""
        # Graph-free metrics stay zero; still log current turn
        info['agent_room_id'] = getattr(self.agent, 'room_id', None)
        turn_log = ExplorationTurnLog(
            **self._update_exp_summary(),
            is_redundant=False,
            action_info=deepcopy(info),
            room_state=self.exploration_room.copy(),
            agent_state=self.agent.copy()
        )
        self.turn_logs.append(turn_log)
        self.history.append(action_sequence)
    
    def _update_exp_summary(self) -> Dict[str, Any]:
        """Calculate current exploration efficiency."""
        # Graph-free: return safe defaults
        self.exp_summary = {
            "coverage": 0.0,
            "redundancy": 0.0,
            "n_valid_queries": 0,
            "n_redundant_queries": 0,
        }
        return self.exp_summary
    

if __name__ == "__main__":
    print("ExplorationManager is graph-free. No tests to run here.")