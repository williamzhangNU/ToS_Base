from typing import Dict, List, Tuple, Optional, Union, Literal
import numpy as np
import copy
from dataclasses import dataclass

from .object import Object
from .relationship import (
    DirPair,
    Dir,
    DirectionRel,
)

@dataclass
class Matrices:
    """Data structure to hold adjacency matrices and their working copies."""
    vertical: np.ndarray
    horizontal: np.ndarray
    vertical_working: np.ndarray
    horizontal_working: np.ndarray
    asked: np.ndarray


class DirectionalGraph:
    """
    Represents spatial relationships between objects in a room using graph.
    
    The class maintains two matrices for tracking vertical and horizontal relationships
    between objects, along with working copies used during updates.

    Two kinds of matrices are used:
    - Final matrices: 
        - (A, B) = 1 means positive direction (front/right)
        - (A, B) = -1 means negative direction (back/left)
        - (A, B) = 0 means same position
    - Working matrices:
        - (A, B) = 1 means there is a directed path from A -> B
        - (A, B) = 0 means no directed path from A -> B
    """
    
    UNKNOWN_VALUE: float = np.nan
    VALID_ROTATION_DEGREES: Tuple[int, ...] = (0, 90, 180, 270)

    def __init__(self, objects: List[Object], is_explore: bool = False, keep_original: bool = False) -> None:
        """
        Initialize adjacency matrices for a set of objects.

        Args:
            objects: List of objects to track relationships between
            is_explore: If True, initialize empty matrices, otherwise, all known
            keep_original: If True, keep the original object if it is moved
        """
        self.size = len(objects) # track valid objects in the graph
        
        if not is_explore:
            matrices = self._init_matrices(objects)
        else:
            matrices = self._create_empty_matrices(self.size)
            
        self._v_matrix = matrices.vertical
        self._h_matrix = matrices.horizontal
        self._v_matrix_working = matrices.vertical_working
        self._h_matrix_working = matrices.horizontal_working
        self._asked_matrix = matrices.asked
        
        self.is_explore = is_explore
        self.keep_original = keep_original

    @classmethod
    def _if_unknown(cls, matrix_value: Union[int, float, np.ndarray]):
        return np.isnan(matrix_value)

    def _create_empty_matrices(self, size: int) -> Matrices:
        """Create empty matrices initialized with unknown values."""
        v_matrix = np.full((size, size), self.UNKNOWN_VALUE)
        h_matrix = np.full((size, size), self.UNKNOWN_VALUE)
        v_matrix_working = np.zeros((size, size))
        h_matrix_working = np.zeros((size, size))
        asked = np.zeros((size, size), dtype=bool)

        # Set diagonal elements
        np.fill_diagonal(v_matrix, 0)
        np.fill_diagonal(h_matrix, 0)
        np.fill_diagonal(v_matrix_working, 1)
        np.fill_diagonal(h_matrix_working, 1)
        np.fill_diagonal(asked, 1)

        return Matrices(v_matrix, h_matrix, v_matrix_working, h_matrix_working, asked)

    @classmethod
    def _dir_to_val(cls, dir: Dir, axis: Literal['vertical', 'horizontal']) -> float:
        """Convert a Dir enum to its matrix value representation."""
        if dir == Dir.UNKNOWN:
            return cls.UNKNOWN_VALUE
            
        mapping = {
            'vertical': {
                Dir.SAME: 0,
                Dir.FORWARD: 1,
                Dir.BACKWARD: -1
            },
            'horizontal': {
                Dir.SAME: 0,
                Dir.RIGHT: 1,
                Dir.LEFT: -1
            }
        }
        return mapping[axis].get(dir, cls.UNKNOWN_VALUE)

    @classmethod
    def _val_to_dir(cls, value: float, axis: Literal['vertical', 'horizontal']) -> Dir:
        """Convert a matrix value to its Dir enum representation."""
        if cls._if_unknown(value):
            return Dir.UNKNOWN
            
        mapping = {
            'vertical': {
                0: Dir.SAME,
                1: Dir.FORWARD,
                -1: Dir.BACKWARD
            },
            'horizontal': {
                0: Dir.SAME,
                1: Dir.RIGHT,
                -1: Dir.LEFT
            }
        }
        return mapping[axis].get(int(value), Dir.UNKNOWN)

    def _init_matrices(self, objects: List[Object]) -> Matrices:
        """
        Initialize adjacency matrices based on object positions.
        One vertical matrix and one horizontal matrix, each for vertical and horizontal relationships
        For working matrices, (A, B) = 1 means there is a directed path from A -> B
        """
        matrices = self._create_empty_matrices(len(objects))
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue
                dir_pair = DirectionRel.get_direction(obj1.pos, obj2.pos)
                matrices.vertical[i, j] = self._dir_to_val(dir_pair.vert, 'vertical')
                matrices.horizontal[i, j] = self._dir_to_val(dir_pair.horiz, 'horizontal')

        # Update working matrices
        matrices.vertical_working[(matrices.vertical == 1) | (matrices.vertical == 0)] = 1
        matrices.horizontal_working[(matrices.horizontal == 1) | (matrices.horizontal == 0)] = 1
        matrices.asked = np.ones((len(objects), len(objects)), dtype=bool)

        return matrices

    def _update_connected_components(self, obj1_id: int, obj2_id: int, matrix: np.ndarray) -> None:
        """
        Update connected components after adding an edge.
        If obj1 -> obj2, update all C connected to D by C -> obj1 -> obj2 -> D
        """
        connected_to_obj1 = matrix[:, obj1_id] == 1 # all objects O1: O1 -> obj1
        connected_from_obj2 = matrix[obj2_id] == 1 # all objects O2: obj2 -> O2
        matrix[np.ix_(connected_to_obj1, connected_from_obj2)] = 1 # all objects O1 -> obj1 -> obj2 -> O2

    def _update_matrices_from_working(self) -> None:
        """Update final matrices based on working matrices state.
        1. For working matrix, if (A, B) = 1 (A -> B)
            - For working matrix, (B, A) = 1 => update (A, B) = 0 in final matrix (A -> B and B -> A)
            - For working matrix, (B, A) = 0 => update (A, B) = 1 in final matrix (only A -> B)  
        """

        unknown_mask = (self._v_matrix_working + self._v_matrix_working.T) == 0 # unknown if both 0
        self._v_matrix = self._v_matrix_working + self._v_matrix_working.T * (-1)
        self._v_matrix[unknown_mask] = self.UNKNOWN_VALUE

        unknown_mask = (self._h_matrix_working + self._h_matrix_working.T) == 0
        self._h_matrix = self._h_matrix_working + self._h_matrix_working.T * (-1)
        self._h_matrix[unknown_mask] = self.UNKNOWN_VALUE
        
    @staticmethod
    def create_graph_from_coordinates(coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a graph from a list of coordinates
        """
        
        # Initialize matrices
        _v_matrix = np.zeros((len(coordinates), len(coordinates)), dtype=np.int8)
        _h_matrix = np.zeros((len(coordinates), len(coordinates)), dtype=np.int8)
        
        # Fill matrices based on coordinates
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i == j:
                    continue
                dir_pair = DirectionRel.get_direction(coordinates[i], coordinates[j])
                _v_matrix[i, j] = DirectionalGraph._dir_to_val(dir_pair.vert, 'vertical')
                _h_matrix[i, j] = DirectionalGraph._dir_to_val(dir_pair.horiz, 'horizontal')
        
        return _v_matrix, _h_matrix
        
        
        # create a new graph with the same size as the original graph
        

    def add_edge(self, obj1_id: int, obj2_id: int, dir_pair: DirPair) -> None:
        """
        Add a directional edge between two objects and update all inferrable relationships.
        
        Args:
            obj1_id: ID of the first object
            obj2_id: ID of the second object
            dir_pair: Direction pair of (horizontal, vertical)
        """
        assert self.is_explore, "Cannot add edges when is_explore is False"

        h_rel, v_rel = dir_pair.horiz, dir_pair.vert
        v_value = self._dir_to_val(v_rel, 'vertical')
        h_value = self._dir_to_val(h_rel, 'horizontal')
        if self._if_unknown(v_value) or self._if_unknown(h_value):
            raise ValueError("Direction must be known")
        
        # determine the newly added edge is a novel query or not
        novel_query = self._if_unknown(self._v_matrix[obj1_id, obj2_id]) or self._if_unknown(self._h_matrix[obj1_id, obj2_id])

        # Update vertical relationships
        if v_value <= 0:
            self._update_connected_components(obj2_id, obj1_id, self._v_matrix_working)
        if v_value >= 0:
            self._update_connected_components(obj1_id, obj2_id, self._v_matrix_working)

        # Update horizontal relationships
        if h_value <= 0:
            self._update_connected_components(obj2_id, obj1_id, self._h_matrix_working)
        if h_value >= 0:
            self._update_connected_components(obj1_id, obj2_id, self._h_matrix_working)

        self._update_matrices_from_working()

        if obj1_id < self.size and obj2_id < self.size:
            self._asked_matrix[obj1_id, obj2_id] = 1
            self._asked_matrix[obj2_id, obj1_id] = 1

        return novel_query

    def add_partial_edge(self, obj1_id: int, obj2_id: int, dir_pair: DirPair) -> None:
        """
        Add a partial directional edge with at least one unknown direction.
        NOTE: the relationship is inferred
        
        Args:
            obj1_id: ID of the first object
            obj2_id: ID of the second object  
            dir_pair: Direction pair with at least one unknown direction
        """
        assert self.is_explore, "Cannot add edges when is_explore is False"
        
        h_rel, v_rel = dir_pair.horiz, dir_pair.vert
        v_value = self._dir_to_val(v_rel, 'vertical')
        h_value = self._dir_to_val(h_rel, 'horizontal')
        
        # Validate that at least one direction is unknown
        if not (self._if_unknown(v_value) or self._if_unknown(h_value)):
            raise ValueError("At least one direction must be unknown")
        
        # Update vertical relationships only if known
        if not self._if_unknown(v_value):
            if v_value <= 0:
                self._update_connected_components(obj2_id, obj1_id, self._v_matrix_working)
            if v_value >= 0:
                self._update_connected_components(obj1_id, obj2_id, self._v_matrix_working)
        
        # Update horizontal relationships only if known
        if not self._if_unknown(h_value):
            if h_value <= 0:
                self._update_connected_components(obj2_id, obj1_id, self._h_matrix_working)
            if h_value >= 0:
                self._update_connected_components(obj1_id, obj2_id, self._h_matrix_working)
        
        self._update_matrices_from_working()

    def add_node(self, obj_anchor_id: int, dir_pair: DirPair) -> None:
        """
        Add a new object to the graph: new_obj is dir_pair relative to obj_anchor_id
        """
        assert self.is_explore, "Cannot add node when is_explore is False"

        # Create expanded matrices
        new_size = self._v_matrix_working.shape[0] + 1
        self.size += 1
        new_obj_id = self.size - 1
        
        v_expanded = np.zeros((new_size, new_size))
        h_expanded = np.zeros((new_size, new_size))
        
        # Copy existing data
        v_expanded[:-1, :-1] = self._v_matrix_working
        h_expanded[:-1, :-1] = self._h_matrix_working
        
        # Set diagonal elements for new position
        v_expanded[-1, -1] = 1
        h_expanded[-1, -1] = 1

        
        # Update working matrices
        self._v_matrix_working = v_expanded
        self._h_matrix_working = h_expanded
        self._update_matrices_from_working()

        # Update asked matrix
        new_asked_matrix = np.zeros((new_size, new_size), dtype=bool)
        new_asked_matrix[:-1, :-1] = self._asked_matrix
        self._asked_matrix = new_asked_matrix
        np.fill_diagonal(self._asked_matrix, 1)
        
        # Update relationships
        self.add_edge(new_obj_id, obj_anchor_id, dir_pair)

    def move_node(self, obj1_id: int, obj2_id: int, dir_pair: DirPair) -> None:
        """
        Move an object (obj1) to a new position relative to another object (obj2).
        
        This method:
        1. Expands matrices to accommodate the new position
        2. Switches rows and columns for the moved object
        3. Updates relationships using the add_edge method
        
        Args:
            obj1_id: ID of the object being moved
            obj2_id: ID of the reference object
            dir_pair: Direction pair of (horizontal, vertical) specifying new relative position
        """
        assert self.is_explore, "Cannot move node when is_explore is False"

        # Create expanded matrices
        new_size = self._v_matrix_working.shape[0] + 1
        v_expanded = np.zeros((new_size, new_size))
        h_expanded = np.zeros((new_size, new_size))
        
        # Copy existing data
        v_expanded[:-1, :-1] = self._v_matrix_working
        h_expanded[:-1, :-1] = self._h_matrix_working
        
        # Set diagonal elements for new position
        v_expanded[-1, -1] = 1
        h_expanded[-1, -1] = 1
        
        # Switch rows and columns for the moved object
        for matrix in [v_expanded, h_expanded]:
            # Switch rows
            matrix[[obj1_id, -1], :] = matrix[[-1, obj1_id], :]
            # Switch columns
            matrix[:, [obj1_id, -1]] = matrix[:, [-1, obj1_id]]
        
        # Update working matrices
        self._v_matrix_working = v_expanded
        self._h_matrix_working = h_expanded
        self._update_matrices_from_working()
        # Update asked matrix
        self._asked_matrix[obj1_id, :] = 0
        self._asked_matrix[:, obj1_id] = 0

        if obj1_id == obj2_id:
            obj2_id = new_size - 1
        
        # Update relationships
        self.add_edge(obj1_id, obj2_id, dir_pair)

        if not self.keep_original:
            self._v_matrix_working = self._v_matrix_working[:-1, :-1]
            self._h_matrix_working = self._h_matrix_working[:-1, :-1]
            self._update_matrices_from_working()

        

    def rotate_axis(self, degree: int) -> None:
        """
        Rotate the coordinate system clockwise by the specified degree.
        
        Args:
            degree: Rotation angle (90, 180, or 270 degrees)
            
        Raises:
            ValueError: If degree is not valid or is_explore is False
        """
        assert self.is_explore, "Cannot rotate when is_explore is False"
        assert degree in self.VALID_ROTATION_DEGREES, f"Degree must be one of {self.VALID_ROTATION_DEGREES}"

        v_working = copy.deepcopy(self._v_matrix_working)
        h_working = copy.deepcopy(self._h_matrix_working)

        rotation_maps = {
            0: (v_working, h_working),
            90: (h_working, v_working.T),
            180: (v_working.T, h_working.T),
            270: (h_working.T, v_working)
        }
        
        self._v_matrix_working, self._h_matrix_working = rotation_maps[degree]
        self._update_matrices_from_working()

    def get_direction(self, obj1_id: int, obj2_id: int) -> DirPair: # NOTE old name: get_dir_rel
        """
        Get the directional relationship between two objects, may be unknown
        
        Args:
            obj1_id: ID of the first object
            obj2_id: ID of the second object
            
        Returns:
            Direction Pair (DirPair)
        """
        v_value = self._v_matrix[obj1_id, obj2_id]
        h_value = self._h_matrix[obj1_id, obj2_id]
        return DirPair(
            horiz=self._val_to_dir(h_value, 'horizontal'),
            vert=self._val_to_dir(v_value, 'vertical')
        )
    
    def get_unknown_pairs(self) -> List[Tuple[int, int]]:
        """
        Get all pairs of objects with unknown spatial relationships.
        
        Returns:
            List of tuples containing (obj1_id, obj2_id) where at least one
            dimension (vertical or horizontal) of their relationship is unknown.
            Only returns unique pairs (no duplicates or self-pairs).
        """
        unknown_v_masks = self._if_unknown(self._v_matrix)
        unknown_h_masks = self._if_unknown(self._h_matrix)
        unknown_masks = unknown_v_masks | unknown_h_masks
        unknown_masks = unknown_masks[:self.size, :self.size] # NOTE ignore original object before movement
        
        # Use numpy to find indices where relationships are unknown
        unknown_indices = np.where(np.triu(unknown_masks, k=1))
        unknown_pairs = [(i.item(), j.item()) for i,j in zip(unknown_indices[0], unknown_indices[1])]
        
        return unknown_pairs
    
    def get_inferable_pairs(self) -> List[Tuple[int, int]]:
        """
        Get all pairs of objects with known spatial relationships.
        
        Returns:
            List of tuples containing (obj1_id, obj2_id) where both
        """
        # first calculate known masks
        known_v_masks = ~self._if_unknown(self._v_matrix)
        known_h_masks = ~self._if_unknown(self._h_matrix)
        known_masks = known_v_masks & known_h_masks
        known_masks = known_masks[:self.size, :self.size] # NOTE ignore original object before movement
        
        inferable_masks = known_masks & ~self._asked_matrix
        inferable_indices = np.where(np.triu(inferable_masks, k=1))
        inferable_pairs = [(i.item(), j.item()) for i,j in zip(inferable_indices[0], inferable_indices[1])]
        
        return inferable_pairs
        
        

    def to_dict(self) -> Dict[str, List[List[float]]]:
        """Convert matrices to a dictionary representation."""
        return {
            'v_matrix': self._v_matrix.tolist(),
            'h_matrix': self._h_matrix.tolist(),
            'v_matrix_working': self._v_matrix_working.tolist(),
            'h_matrix_working': self._h_matrix_working.tolist(),
            'asked_matrix': self._asked_matrix.tolist(),
            'size': self.size,
            'is_explore': self.is_explore,
            'keep_original': self.keep_original
        }
    
    @classmethod
    def from_dict(cls, graph_dict: Dict[str, List[List[float]]]) -> 'DirectionalGraph':
        """Convert a dictionary representation back to a DirectionalGraph object."""
        instance = cls(objects=[], is_explore=True)
        instance._v_matrix = np.array(graph_dict['v_matrix'])
        instance._h_matrix = np.array(graph_dict['h_matrix'])
        instance._v_matrix_working = np.array(graph_dict['v_matrix_working'])
        instance._h_matrix_working = np.array(graph_dict['h_matrix_working'])
        instance._asked_matrix = np.array(graph_dict['asked_matrix'])
        instance.size = graph_dict['size']
        instance.is_explore = graph_dict['is_explore']
        instance.keep_original = graph_dict['keep_original']
        return instance
    
    def copy(self) -> 'DirectionalGraph':
        """
        Create a deep copy of the DirectionalGraph.
        
        Returns:
            A new DirectionalGraph object with identical matrices.
        """
        # Create a new instance with the same size and is_explore setting
        new_adj_mat = object.__new__(DirectionalGraph)
        new_adj_mat.size = self.size
        new_adj_mat.is_explore = self.is_explore
        new_adj_mat.keep_original = self.keep_original
        
        # Copy the matrices
        new_adj_mat._v_matrix = copy.deepcopy(self._v_matrix)
        new_adj_mat._h_matrix = copy.deepcopy(self._h_matrix)
        new_adj_mat._v_matrix_working = copy.deepcopy(self._v_matrix_working)
        new_adj_mat._h_matrix_working = copy.deepcopy(self._h_matrix_working)
        new_adj_mat._asked_matrix = copy.deepcopy(self._asked_matrix)
        return new_adj_mat
