import copy
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from ..core.object import Agent
from ..actions import *
from ..core.room import Room
from .spatial_solver import SpatialSolver

@dataclass
class ExplorationTurnLog:
    """Log data for a single exploration turn."""
    node_coverage: float
    edge_coverage: float
    step: int
    action_counts: Dict[str, int]
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None
    information_gain: Optional[Dict[str, Any]] = None  # Information gain metrics
    exploration_quality: Optional[float] = None

    def to_dict(self):
        return {
            "node_coverage": self.node_coverage,
            "edge_coverage": self.edge_coverage,
            "step": self.step,
            "action_counts": dict(self.action_counts),
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "information_gain": self.information_gain or 0.0,
            "exploration_quality": self.exploration_quality or 0.0,
        }

class ExplorationManager:
    """Minimal exploration manager without graphs.

    - Keeps copies of `room` and `agent` for simulation.
    - Executes actions and logs turns.
    - Graph-related metrics default to safe zeros.
    """
    DEFAULT_EXP_SUMMARY = {"node_coverage": 0.0, "edge_coverage": 0.0, "n_exploration_steps": 0, "action_counts": {}}
    
    def __init__(self, room: Room, agent: Agent, enable_information_gain: bool = False, grid_size: int | None = None, enable_exploration_quality: bool = True):
        self.base_room = room.copy()
        self.exploration_room = room.copy()
        self.agent = agent.copy()
        self.keep_object_names = [self.agent.name] + [obj.name for obj in getattr(self.exploration_room, 'all_objects', [])]

        self.exp_summary = copy.deepcopy(self.DEFAULT_EXP_SUMMARY)
        self.turn_logs: List[ExplorationTurnLog] = []
        # History now stores ActionResult for each executed action (in order)
        self.history: List['ActionResult'] = []
        
        # Coverage tracking (exclude gates)
        self._init_node_name = "initial_pos"
        self.init_pos = self.agent.init_pos.copy()
        self._init_room_id = int(self.agent.init_room_id)

        # Node names: all objects in the exploration room
        self.node_names: List[str] = [o.name for o in self.exploration_room.all_objects]

        # Edge targets: per-room object pairs + (init, object-in-init-room)
        self.target_edges: Set[frozenset] = set()
        for rid, names in self.exploration_room.objects_by_room.items():
            names += self.exploration_room.gates_by_room.get(rid, [])
            if not names:
                continue
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    self.target_edges.add(frozenset({a, b}))
        for name in self.exploration_room.objects_by_room[self._init_room_id] + self.exploration_room.gates_by_room.get(self._init_room_id, []):
            self.target_edges.add(frozenset({self._init_node_name, name}))
        
        self.observed_nodes: Set[str] = set()
        self.known_edges: Set[frozenset] = set()

        # Action counts and costs
        self.action_counts: Dict[str, int] = {}
        self.action_cost: int = 0
        # Observed names (objects and gates) to gate Move() eligibility
        self.observed_items: Set[str] = set()
        
        # Information gain control
        self.enable_information_gain = bool(enable_information_gain)
        # Exploration quality control (per-turn when enabled)
        self.enable_exploration_quality = bool(enable_exploration_quality)
        # Grid size for solver metrics (use provided or infer from mask; fallback 10)
        inferred_g = (max(self.exploration_room.mask.shape) if getattr(self.exploration_room, 'mask', None) is not None else 10)
        self.grid_size: int = int(inferred_g if grid_size is None else grid_size)
        # Initialize spatial solver for information gain tracking (only when enabled)
        if self.enable_information_gain:
            object_names = self.node_names + ['initial_pos']
            self.spatial_solver = SpatialSolver(object_names, self.grid_size)
            self.spatial_solver.set_initial_position('initial_pos', (0, 0))
            counts = self.spatial_solver.get_num_possible_positions()
            self.previous_total_positions = sum(counts.values())
        else:
            self.spatial_solver = None
            self.previous_total_positions = 0
        
    def _execute_and_update(self, action: BaseAction, **kwargs) -> ActionResult:
        """Execute action and update exploration state."""
        # Enforce "observed-before-move"
        if isinstance(action, MoveAction):
            kwargs['observed_items'] = list(self.observed_items)
        result = action.execute(self.exploration_room, self.agent, **kwargs)
        # Log every action result to history immediately
        self.history.append(result)
        if not result.success:
            return result
        
        # Count action, cost, and update coverage
        self.action_counts[result.action_type] = self.action_counts.get(result.action_type, 0) + 1
        self.action_cost += int(action.cost)
        if isinstance(action, ObserveAction):
            self._update_coverage_from_observe(result)
        
        return result



    def execute_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result."""
        return self._execute_and_update(action)
    
    def execute_success_action(self, action: BaseAction, **kwargs) -> ActionResult:
        """Execute single action and return result (must be successful)."""
        result = self._execute_and_update(action, **kwargs)
        assert result.success, f"Action {action} with kwargs {kwargs} failed: {result.message}"
        return result

    def execute_action_sequence(self, action_sequence: ActionSequence) -> Tuple[Dict[str, Any], List[ActionResult]]:
        """
        Execute a sequence of motion actions followed by a final action.
        If any motion action fails, execute an observe action and end.
        Returns info and list of action results.
        """
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
                obs_result.message = f"Subsequent actions are skipped due to failure, instead an observe is executed: {obs_result.message}"
                action_results.append(obs_result)
                assert obs_result.success, f"Observe action failed: {obs_result.message}"
                info.update(obs_result.data)
                self._log_exploration(action_sequence, action_results)
                return info, action_results

        # Execute final action
        final_action = action_sequence.final_action
        result = self._execute_and_update(final_action)
        action_results.append(result)
        assert result.success, f"Final action {final_action} failed: {result.message}"
        info.update(result.data)

        # Always log before return
        self._log_exploration(action_sequence, action_results)
        return info, action_results
    
    def finish_exploration(self, return_to_origin: bool = True) -> Room:
        """Complete exploration and return final room state."""
        if return_to_origin:
            result = self.execute_action(ReturnAction())
            if not result.success:
                raise ValueError(f"Failed to return to origin: {result.message}")
        return self.exploration_room
    
    def get_exp_summary(self) -> Dict[str, Any]:
        """Get exploration summary."""
        return dict(self._update_exp_summary())
    
    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict] = None) -> Dict[str, Any]:
        """Calculate exploration performance for a group from env_data_list."""
        if not env_data_list:
            return {"avg_coverage": 0.0, "avg_exploration_steps": 0.0, "avg_node_coverage": 0.0, "avg_edge_coverage": 0.0}

        # Calculate metrics from last exploration log of each sample
        node_coverages = []
        edge_coverages = []
        exploration_steps = []

        for env_data in env_data_list:
            env_turn_logs = env_data.get('env_turn_logs', [])

            # Find the last exploration turn that has actual exploration log data
            last_exploration_log = None
            for turn_log in reversed(env_turn_logs):
                if turn_log.get('is_exploration_phase', False):
                    exploration_log = turn_log.get('exploration_log', {})
                    if exploration_log:  # Only use if exploration_log is not empty
                        last_exploration_log = exploration_log
                        break

            if last_exploration_log:
                node_coverages.append(last_exploration_log.get('node_coverage', 0.0))
                edge_coverages.append(last_exploration_log.get('edge_coverage', 0.0))
                exploration_steps.append(last_exploration_log.get('step', 0))

        n = len(node_coverages) if node_coverages else 1
        result = {}

        # Only add metrics to result if they have valid data
        if exploration_steps:
            result["avg_exploration_steps"] = sum(exploration_steps) / n
        if node_coverages:
            result["avg_node_coverage"] = sum(node_coverages) / n
        if edge_coverages:
            result["avg_edge_coverage"] = sum(edge_coverages) / n

        # Calculate average infogain per turn across all samples
        infogain_per_turn = ExplorationManager._calculate_infogain_per_turn(env_data_list)
        if infogain_per_turn is not None:
            result["infogain_per_turn"] = infogain_per_turn

        return result
    
    @staticmethod
    def _calculate_infogain_per_turn(env_data_list: List[Dict]) -> List[float]:
        """Calculate average information gain for each turn across all samples."""
        # Collect all turn information gains by turn index
        turn_infogains = defaultdict(list)  # turn_index -> list of infogain values
        PAD = 0.0
        
        for env_data in env_data_list:
            env_turn_logs = env_data.get('env_turn_logs', [])
            for turn_idx, turn_log in enumerate(env_turn_logs):
                # Only consider exploration phases
                if turn_log.get('is_exploration_phase', False):
                    infogain = turn_log.get('exploration_log', {}).get('information_gain')
                    if infogain is not None:
                        turn_infogains[turn_idx].append(infogain)
        
        # Calculate averages for each turn
        max_turns = max(turn_infogains.keys()) if turn_infogains else -1
        avg_infogains = []
        
        for turn_idx in range(max_turns + 1):
            if turn_idx in turn_infogains and turn_infogains[turn_idx]:
                avg_infogain = sum(turn_infogains[turn_idx]) / len(turn_infogains[turn_idx])
                avg_infogains.append(avg_infogain)
            else:
                avg_infogains.append(PAD)
        
        return avg_infogains
    
    # No passive history generation here; proxies produce text histories directly.
    
    # === Coverage helpers ===
    def _anchor_name(self) -> Optional[str]:
        # If standing on an object position, use that object as anchor (exclude gates)
        for obj in self.exploration_room.all_objects:
            if np.allclose(obj.pos, self.agent.pos):
                return obj.name
        # Initial position anchor
        if np.allclose(self.agent.pos, self.init_pos):
            return self._init_node_name
        raise ValueError("No anchor found")

    def _update_coverage_from_observe(self, observe_result: 'ActionResult') -> None:
        visible = observe_result.data.get('visible_objects', []) or []
        # node coverage
        for name in visible:
            self.observed_items.add(name)
            if name in self.node_names:
                self.observed_nodes.add(name)
        # edge coverage: observe A from B (B is anchor)
        anchor = self._anchor_name()
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


    
    def _log_exploration(self, action_sequence: ActionSequence, action_results: List['ActionResult']) -> None:
        """Log exploration history and efficiency."""
        # Calculate total information gain ratio for this turn (optional)
        information_gain_ratio = None
        if self.enable_information_gain:
            for action_result in action_results:
                if action_result.action_type in ('observe', 'query'):
                    information_gain_ratio = self._calculate_single_action_information_gain(action_result)
        else:
            information_gain_ratio = 0.0

        # Per-turn exploration quality (optional)
        turn_quality = self._compute_exploration_quality() if self.enable_exploration_quality else 0.0
        
        # Log current turn with coverage snapshot
        self._update_exp_summary()
        step_idx = len(self.turn_logs) + 1
        turn_log = ExplorationTurnLog(
            node_coverage=self.exp_summary.get('node_coverage', 0.0),
            edge_coverage=self.exp_summary.get('edge_coverage', 0.0),
            step=step_idx,
            action_counts=dict(self.exp_summary.get('action_counts', {})),
            room_state=self.exploration_room.copy(),
            agent_state=self.agent.copy(),
            information_gain=information_gain_ratio if information_gain_ratio is not None else (self.turn_logs[-1].information_gain if self.turn_logs else 0.0),
            exploration_quality=turn_quality
        )
        self.turn_logs.append(turn_log)
    
    def _update_exp_summary(self) -> Dict[str, Any]:
        """Calculate current coverage and summary stats."""
        node_cov = len(self.observed_nodes) / len(self.node_names)
        edge_cov = len(self.known_edges) / len(self.target_edges)
        info_gain_list = [turn_log.information_gain for turn_log in self.turn_logs] if self.turn_logs else []
        acc_info_gain = sum(info_gain_list)
        avg_info_gain = acc_info_gain / len(self.turn_logs) if self.turn_logs else 0.0
        # Latest exploration quality (optional, mirrors per-turn computation)
        quality = self._compute_exploration_quality() if self.enable_exploration_quality else None
        self.exp_summary = {
            "node_coverage": node_cov,
            "edge_coverage": edge_cov,
            "n_exploration_steps": len(self.turn_logs),
            "action_counts": dict(self.action_counts),
            "action_cost": int(self.action_cost),
            "exploration_cost": int(self.action_cost),
            "info_gain_list": info_gain_list,
            "acc_info_gain": acc_info_gain,
            "avg_info_gain": avg_info_gain,
            "exploration_quality": quality,
        }
        return self.exp_summary
    
    def _calculate_single_action_information_gain(self, action_result: 'ActionResult') -> float:
        """Calculate information gain as negative log of ratio between current and previous total positions."""
        if not self.enable_information_gain or (self.spatial_solver is None):
            return 0.0
        # Store previous positions before processing the action
        previous_positions = self.previous_total_positions
        
        # Only process observation actions that have relation triples
        if action_result.action_type in ('observe', 'query'):
            triples = action_result.data.get('relation_triples', []) if hasattr(action_result, 'data') else []
            if triples:
                # Add observations to spatial solver
                self.spatial_solver.add_observation(triples)
        
        # Calculate position count after action
        counts = self.spatial_solver.get_num_possible_positions()
        current_positions = sum(counts.values())
        
        # Update previous_total_positions for next calculation
        self.previous_total_positions = current_positions
        
        # Calculate and return negative log of the ratio
        if previous_positions > 0:
            ratio = current_positions / previous_positions
            return -np.log(ratio) if ratio > 0 else 0.0
        else:
            return 0.0

    # === Exploration quality helpers ===
    def _full_grid_cell_count(self) -> int:
        return int(self.grid_size) * int(self.grid_size)

    def _final_position_counts(self) -> Dict[str, int]:
        """Counts of possible positions per variable at the end of exploration.
        Uses existing solver if available, otherwise rebuilds a solver from history.
        """
        if self.spatial_solver is not None:
            return self.spatial_solver.get_num_possible_positions()
        # Build a temporary solver and ingest history triples
        solver = SpatialSolver(self.node_names + ['initial_pos'], self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))
        for ar in self.history:
            try:
                if getattr(ar, 'action_type', None) in ('observe', 'query'):
                    triples = ar.data.get('relation_triples', []) if hasattr(ar, 'data') else []
                    if triples:
                        solver.add_observation(triples)
            except Exception:
                continue
        return solver.get_num_possible_positions()

    def _compute_exploration_quality(self) -> float | None:
        """Compute quality = sum_i log2(M/Ci) / (N * log2(M)). Exclude 'initial_pos'. Include gates.
        Returns None if computation is not applicable.
        """
        try:
            counts = self._final_position_counts()
            M = self._full_grid_cell_count()
            if M <= 1:
                return 0.0
            names = [n for n in counts.keys() if n != 'initial_pos']
            if not names:
                return 0.0
            denom = len(names) * np.log2(M)
            if denom <= 0:
                return 0.0
            total = 0.0
            for n in names:
                Ci = max(1, int(counts.get(n, M)))
                total += float(np.log2(M / Ci))
            return float(total / denom)
        except Exception:
            return None

if __name__ == "__main__":
    pass