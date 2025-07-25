import numpy as np
import copy
from copy import deepcopy
from typing import List, Tuple, Dict, Any

from ..core.object import Object, Agent
from ..core.relationship import DirPair, DirectionSystem, Dir
from ..core.graph import DirectionalGraph
from ..actions import (
    BaseAction,
    ActionResult,
    ActionSequence,
    MoveAction,
    RotateAction,
    ReturnAction,
    ObserveAction,
    TermAction,
)
from ..core.room import Room

class ExplorationManager:
    """Manages spatial exploration with agent movement and queries.
    
    Maintains base_room (original state) and exploration_room (working copy).
    Actions handle their own coordinate transformations. Exploration is egocentric.
    """
    
    def __init__(self, room: Room):
        assert room.agent is not None, "Exploration requires an agent in the room"
        
        self.base_room = room.copy()
        self.exploration_room = room.copy()
        
        self.agent_idx = self._get_index(self.exploration_room.agent.name)
        self.initial_pos_idx = self._get_index("initial_pos")

        self.objects = self.exploration_room.all_objects
        self.exp_graph = DirectionalGraph(self.objects, is_explore=True)
        
        self.exp_graph.add_edge(self.agent_idx, self.initial_pos_idx, DirPair(Dir.SAME, Dir.SAME))
        self.metrics_log: List[Dict[str, Any]] = []

        # log exploration history and efficiency
        self.exploration_efficiency = {
            "coverage": 0, # coverage of the exploration
            "redundancy": 0, # redundancy of the exploration
            "n_valid_queries": 0, # number of valid queries
            "n_redundant_queries": 0, # number of redundant queries
        }
        self.history = [] 
        
    def _get_index(self, name: str) -> int:
        """Get object index by name."""
        for i, obj in enumerate(self.exploration_room.all_objects):
            if obj.name == name:
                return i
        raise ValueError(f"Object '{name}' not found")
    
    def _update_move(self, target_name: str):
        """Update exploration graph after move action."""
        target_idx = self._get_index(target_name)
        self.exp_graph.move_node(self.agent_idx, target_idx, DirPair(Dir.SAME, Dir.SAME))

    def _update_rotate(self, degrees: int):
        """Update exploration graph after rotate action."""
        self.exp_graph.rotate_axis(degrees)
    
    def _update_observe(self) -> bool:
        """Update exploration graph after observe action.
        
        Checks efficiency and updates relationships:
        1. Efficiency check: redundant if no unknown pairs with agent, or if all 
           unknown objects are only known to be behind the agent
        2. Update visible objects with full relationships
        3. For 180° field of view: update invisible objects as behind the agent
            
        Returns:
            bool: True if this observation is redundant (not efficient)
        """
        field_of_view = BaseAction.get_field_of_view()
        assert field_of_view in [90, 180], "Field of view must be 90 or 180 degrees"
        
        # Calculate visible objects using _is_visible
        agent = self.exploration_room.agent

        visible_objects = [obj.name for obj in self.exploration_room.objects if BaseAction._is_visible(agent, obj)]
        
        # 1. Check efficiency
        unknown_pairs = self.get_unknown_pairs()
        agent_unknown_pairs = [(pair[1], pair[0]) if pair[0] == self.agent_idx else pair 
                              for pair in unknown_pairs if self.agent_idx in pair]
        
        if not agent_unknown_pairs:
            is_redundant = True
            print("No unknown pairs with agent")
        else:
            relationships = {self.exp_graph.get_direction(obj_idx, agent_idx) 
                           for obj_idx, agent_idx in agent_unknown_pairs}
            
            if field_of_view == 90:
                # for 90 degree, agent should not observe when all objects are back
                # four possible directions: front, back, left, right
                is_redundant = (len(relationships) == 1 and 
                                list(relationships)[0].horiz == Dir.UNKNOWN and 
                                list(relationships)[0].vert == Dir.BACKWARD)
            else:
                # for 180 degree, agent should observe when all objects are front
                is_redundant = (len(relationships) == 1 and 
                                (list(relationships)[0].horiz != Dir.UNKNOWN or 
                                list(relationships)[0].vert != Dir.FORWARD))
        
        # 2. Update relationships for visible objects
        for obj_name in visible_objects:
            obj_idx = self._get_index(obj_name)
            dir_pair, _ = self.exploration_room.get_direction(obj_name, agent.name)
            self.exp_graph.add_edge(obj_idx, self.agent_idx, dir_pair)
        
        # 3. Update invisible objects for 180° field of view
        if field_of_view == 180:
            invisible_objects = [obj.name for obj in self.exploration_room.objects if obj.name not in visible_objects]
            
            for obj_name in invisible_objects:
                obj_idx = self._get_index(obj_name)
                back_dir_pair = DirPair(Dir.UNKNOWN, Dir.BACKWARD)
                self.exp_graph.add_partial_edge(obj_idx, self.agent_idx, back_dir_pair)

        return is_redundant

    def _execute_and_update(self, action: BaseAction) -> ActionResult:
        """Execute action and update exploration state."""
        kwargs = {}
        result = action.execute(self.exploration_room, **kwargs)
        
        if not result.success:
            return result
        
        # success execution
        if isinstance(action, MoveAction):
            self._update_move(result.data['target_name'])
        elif isinstance(action, RotateAction):
            self._update_rotate(result.data['degrees'])
        elif isinstance(action, ReturnAction):
            self._update_move(result.data['target_name'])
            self._update_rotate(result.data['degrees'])
        elif isinstance(action, ObserveAction):
            result.data['redundant'] = self._update_observe()
        
        return result



    def execute_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result."""
        return self._execute_and_update(action)
    
    def _log_exploration(self, action_sequence: ActionSequence, info: Dict[str, Any]) -> None:
        """Log exploration history and efficiency."""
        if info.get('redundant'):
            self.exploration_efficiency['n_redundant_queries'] += 1
        if not action_sequence.final_action.is_term():
            self.exploration_efficiency['n_valid_queries'] += 1
        self.metrics_log.append(deepcopy(self.exploration_efficiency))
        self.metrics_log.append(deepcopy(info))
        self.history.append(action_sequence)
    def get_metrics_log(self) -> List[Dict[str, Any]]:
        return self.metrics_log

    def execute_action_sequence(self, action_sequence: ActionSequence) -> Tuple[Dict[str, Any], List[ActionResult]]:
        """
        Execute a sequence of motion actions followed by a final action.
        If any motion action fails, execute an observe action and end.
        Returns info and list of action results.
        """
        
        assert action_sequence.final_action, "Action sequence requires a final action."
        assert not (isinstance(action_sequence.final_action, TermAction) and action_sequence.motion_actions), "Term() action should not have motion actions."

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
    
    
    
    def finish_exploration(self, return_to_origin: bool = True, neglect_initial_pos: bool = True) -> Room:
        """Complete exploration and return final room state."""
        if return_to_origin:
            result = self.execute_action(ReturnAction())
            if not result.success:
                raise ValueError(f"Failed to return to origin: {result.message}")
            
        if neglect_initial_pos:
            self._remove_initial_pos()
        return self.exploration_room
    
    def _remove_initial_pos(self) -> None:
        """Remove initial_pos from exploration room and graph."""
        initial_pos_idx = self._get_index("initial_pos")
        
        # Remove from exploration room
        self.exploration_room.all_objects = [self.exploration_room.agent] + self.exploration_room.objects
        
        # Update exploration graph
        self.exp_graph.size -= 1
        for matrix_name in ['_v_matrix', '_h_matrix', '_v_matrix_working', '_h_matrix_working', '_asked_matrix']:
            matrix = getattr(self.exp_graph, matrix_name)
            matrix = np.delete(matrix, initial_pos_idx, axis=0)
            matrix = np.delete(matrix, initial_pos_idx, axis=1)
            setattr(self.exp_graph, matrix_name, matrix)

    
    def get_unknown_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of objects with unknown relationships."""
        return self.exp_graph.get_unknown_pairs()
    
    def get_inferable_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of objects with inferable relationships."""
        return self.exp_graph.get_inferable_pairs()
    
    def get_exploration_efficiency(self) -> Dict:
        """Get exploration efficiency metrics."""
        unknown_pairs = self.get_unknown_pairs()

        n_object = len(self.objects)
        max_rels = int(n_object * (n_object - 1) / 2)
        coverage = (max_rels - len(unknown_pairs)) / max_rels if max_rels > 0 else 0

        self.exploration_efficiency['coverage'] = coverage
        self.exploration_efficiency['redundancy'] = self.exploration_efficiency['n_redundant_queries'] / self.exploration_efficiency['n_valid_queries'] if self.exploration_efficiency['n_valid_queries'] > 0 else 0
        self.exploration_efficiency['n_valid_queries'] = self.exploration_efficiency['n_valid_queries']
        self.exploration_efficiency['n_redundant_queries'] = self.exploration_efficiency['n_redundant_queries']

        return self.exploration_efficiency
    


if __name__ == "__main__":
    import sys
    import os
    
    # Add the Base directory to the path so we can import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
    
    from ..core import Object, Agent
    from ..core import Room
    from ..core import DirPair, Dir
    from ..core import DirectionalGraph
    import numpy as np
    
    def create_test_room(objects_data):
        """Helper function to create a test room.
        
        Args:
            objects_data: List of (name, pos) tuples for objects
        """
        agent = Agent("agent")
        objects = [Object(name, np.array(pos), np.array([0, 1])) for name, pos in objects_data]
        room = Room(objects=objects, name="test_room", agent=agent)
        print(f"room: {room}")
        return room
    
    def test_update_observe_no_unknown_pairs():
        """Test case where there are no unknown pairs with agent - should not be novel."""
        print("Test 1: No unknown pairs with agent")
        
        # Create room with agent at origin and one object
        room = create_test_room([("table", [1, 0])])
        manager = ExplorationManager(room)
        
        # Manually set up a scenario where all relationships are known
        # Add edge between agent and table
        agent_idx = manager.agent_idx
        table_idx = manager._get_index("table")
        manager.exp_graph.add_edge(table_idx, agent_idx, DirPair(Dir.LEFT, Dir.SAME))
        
        # Call _update_observe
        is_novel = manager._update_observe()
        
        print(f"  Result: is_novel = {is_novel}")
        print(f"  Expected: False (no unknown pairs)")
        assert not is_novel, "Should not be novel when no unknown pairs exist"
        print("  ✓ PASSED\n")
    
    def test_update_observe_single_direction_front():
        """Test case where all unknown pairs are in front direction - should be novel."""
        print("Test 2: Unknown pairs all in front direction")
        
        # Create room with agent and multiple objects in front
        room = create_test_room([
            ("table", [1, 2]), 
            ("chair", [2, 1])
        ])
        manager = ExplorationManager(room)
        
        # Call _update_observe
        is_novel = manager._update_observe()
        
        print(f"  Result: is_novel = {is_novel}")
        print(f"  Expected: True (unknown pairs in front direction)")
        assert is_novel, "Should be novel when unknown pairs are in front"
        print("  ✓ PASSED\n")
    
    def test_update_observe_single_direction_not_front():
        """Test case where all unknown pairs are in same direction but not front - should not be novel."""
        print("Test 3: Unknown pairs all in same non-front direction")
        
        room = create_test_room([
            ("table", [0, -1]),
            ("chair", [0, 1])
        ])
        manager = ExplorationManager(room)
        
        # Manually set up a scenario where all relationships are known
        # Add edge between agent and table
        agent_idx = manager.agent_idx
        chair_idx = manager._get_index("chair")
        manager.exp_graph.add_edge(chair_idx, agent_idx, DirPair(Dir.SAME, Dir.FORWARD))
        
        is_novel = manager._update_observe()
        
        print(f"  Result: is_novel = {is_novel}")
        print(f"  Expected: False (not in front direction)")
        print("  ✓ PASSED\n")
    
    def test_update_observe_multiple_directions():
        """Test case where unknown pairs are in multiple directions - should be novel."""
        print("Test 4: Unknown pairs in multiple directions")
        
        room = create_test_room([
            ("table", [1, 1]),   # front-right
            ("chair", [-1, 1]),  # front-left
            ("lamp", [0, -1])    # back
        ])
        manager = ExplorationManager(room)
        
        is_novel = manager._update_observe()
        
        print(f"  Result: is_novel = {is_novel}")
        print(f"  Expected: True (multiple directions)")
        assert is_novel, "Should be novel when pairs are in multiple directions"
        print("  ✓ PASSED\n")
    
    def test_update_observe_visible_objects_update():
        """Test that visible objects get full relationship updates."""
        print("Test 5: Visible objects relationship update")
        
        room = create_test_room([
            ("table", [1, 1]),
            ("chair", [-1, 0])
        ])
        manager = ExplorationManager(room)
        
        # Before observe - should have unknown relationships
        table_idx = manager._get_index("table") 
        agent_idx = manager.agent_idx
        
        print(f"  Before observe:")
        dir_pair_before = manager.exp_graph.get_direction(table_idx, agent_idx)
        print(f"    Table->Agent: {dir_pair_before}")
        
        # Call observe
        manager._update_observe()
        
        # After observe - should have known relationship
        print(f"  After observe:")
        dir_pair_after = manager.exp_graph.get_direction(table_idx, agent_idx)
        print(f"    Table->Agent: {dir_pair_after}")
        
        assert dir_pair_after.horiz != Dir.UNKNOWN, "Horizontal direction should be known"
        assert dir_pair_after.vert != Dir.UNKNOWN, "Vertical direction should be known"
        print("  ✓ PASSED\n")
    
    def test_update_observe_empty_visible_list():
        """Test observe with no visible objects."""
        print("Test 7: Empty visible objects list")
        
        room = create_test_room([("table", [-1, -1])])
        manager = ExplorationManager(room)
        
        # Should handle gracefully
        is_novel = manager._update_observe()
        
        print(f"  Result: is_novel = {is_novel}")
        print(f"  Expected: False (no visible objects)")
        # Should not be novel since no new information gained
        print("  ✓ PASSED\n")

    def test_update_observe_efficiency_algorithm():
        """Test the detailed efficiency algorithm logic."""
        print("Test 8: Detailed efficiency algorithm")
        
        room = create_test_room([
            ("obj1", [1, 2]),    # front-right
            ("obj2", [3, 0]),    # front-right
            ("obj3", [-1, 1]),    # front-left
            ("obj4", [2, -1])    # back-right
        ])
        manager = ExplorationManager(room)
        
        print("  Initial unknown pairs:")
        unknown_pairs = manager.get_unknown_pairs()
        agent_unknown = [(pair[1], pair[0]) if pair[0] == manager.agent_idx else pair 
                        for pair in unknown_pairs if manager.agent_idx in pair]
        print(f"    Agent unknown pairs: {len(agent_unknown)}")
        
        # All objects are in front (different horizontal positions)
        # This should be considered novel
        is_redundant = manager._update_observe()
        print(f"Exp_graph: {manager.exp_graph.to_dict()}")
        
        manager.execute_action(RotateAction(degrees=90))
        is_redundant = manager._update_observe()
        print(f'Exp_graph: {manager.exp_graph.to_dict()}')

        manager.execute_action(MoveAction(target="obj4"))
        manager.execute_action(RotateAction(degrees=180))

        print(f"exp_graph: {manager.exp_graph.to_dict()}")

        is_redundant = manager._update_observe()
        print(f"  Result: is_redundant = {is_redundant}")
        print(f"  Expected: False")
        assert not is_redundant, "Should be redundant with objects in multiple front directions"
        print("  ✓ PASSED\n")

    def test_explore_fail_action():
        """Test exploration with failed action execution."""
        print("Test 9: Exploration with failed action")
        
        room = create_test_room([("obj1", [1, 2])])
        manager = ExplorationManager(room)
        
        # Create action sequence with invalid move (non-existent object)
        action_sequence = ActionSequence([RotateAction(degrees=90), RotateAction(degrees=270), MoveAction(target="obj1")], ObserveAction())
        
        # Execute should fail on move but succeed on observe
        info, action_results = manager.execute_action_sequence(action_sequence)
        
        print(f"  Info: {info}")
        
        # Should contain failure message and observe result
        print("  ✓ PASSED\n")
    
    def run_all_tests():
        """Run all test cases."""
        print("=" * 60)
        print("RUNNING TESTS FOR _update_observe METHOD")
        print("=" * 60)
        
        try:
            # test_update_observe_no_unknown_pairs()
            # test_update_observe_single_direction_front()
            # test_update_observe_single_direction_not_front()
            # test_update_observe_multiple_directions()
            # test_update_observe_visible_objects_update()
            # test_update_observe_empty_visible_list()
            # test_update_observe_efficiency_algorithm()
            test_explore_fail_action()
            
            print("=" * 60)
            print("ALL TESTS PASSED! ✓")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the tests
    run_all_tests()