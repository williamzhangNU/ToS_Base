import numpy as np
from typing import List
import sys

from ..core.room import Room
from ..core.constant import CANDIDATE_OBJECTS
from ..core.object import Object, Agent

def generate_room(
        room_range: tuple[int, int],
        n_objects: int,
        candidate_objects: list[Object],
        generation_type: str,
        perspective: str,
        np_random: np.random.Generator,
        room_name: str = 'room',
    ) -> Room:
    """
    Generate a room based on the given configuration
    """
    if generation_type == 'rand' or generation_type == 'pov' or generation_type == 'a2e':
        objects = generate_random_objects(
            n=n_objects,
            candidate_list=candidate_objects,
            random_generator=np_random,
            perspective_taking=(generation_type == 'pov'),
            room_range=room_range,
        )
    elif generation_type == 'rot':
        objects = generate_room_for_rotation_eval(
            n=n_objects,
            candidate_list=candidate_objects,
            random_generator=np_random,
            room_range=room_range,
        )
    # elif generation_type == 'a2e':
    #     objects = generate_allo2ego_objects(
    #         n=n_objects,
    #         candidate_list=candidate_objects,
    #         random_generator=np_random,
    #         room_range=room_range,
    #     )
    else:
        raise ValueError(f"Invalid generation type: {generation_type}")
    
    return Room(
        objects=objects,
        name=room_name,
        agent=Agent() if perspective == 'ego' else None,
    )



def generate_random_objects(
    n: int,
    random_generator: np.random.Generator,
    candidate_list: list[str],
    room_range: list[int] = [-10, 10],
    perspective_taking: bool = False,
) -> list[Object]:
    """Generate random objects with random positions and orientations."""
    # Select random objects from candidate list
    indices = random_generator.choice(len(candidate_list), n, replace=False)
    names = [candidate_list[i] for i in indices]
    
    # Generate random positions ensuring no two objects are at the same position
    positions = [np.array([0, 0])]
    while len(positions) < n + 1:
        pos = random_generator.integers(room_range[0], room_range[1], (1, 2))[0]
        if not any(np.array_equal(pos, existing_pos) for existing_pos in positions):
            positions.append(pos)
    positions = np.array(positions[1:])
    print(f'[DEBUG] positions: {positions}')
    orientations = random_generator.integers(0, 4, n)
    
    # Map orientation values to vectors
    ori_vectors = {
        0: [0, 1],
        1: [1, 0],
        2: [0, -1],
        3: [-1, 0]
    }
    
    # Create and return object list
    objects = []
    for name, pos, ori_idx in zip(names, positions, orientations):
        ori = np.array(ori_vectors[ori_idx] if perspective_taking else [0, 1])
        objects.append(Object(name=name, pos=pos, ori=ori))

    return objects



def generate_room_for_rotation_eval(
    n: int,
    random_generator: np.random.Generator,
    room_range: list[int] = [-10, 10],
    candidate_list: list[str] = CANDIDATE_OBJECTS,
) -> list[Object]:
    """Generate random objects for rotation evaluation with specific placement constraints:
    1. No object directly in front of agent
    2. No object overlap
    3. No two objects in the same half-axis
    4. Objects in the same quadrant must follow specific relative positioning rules
    """
    def _is_valid_placement(objects: list[Object], new_obj: Object) -> bool:
        # Block placement directly in front of agent
        if new_obj.pos[0] == 0 and new_obj.pos[1] > 0:
            return False
            
        for obj in objects:
            # No overlapping positions
            if np.array_equal(new_obj.pos, obj.pos):
                return False
                
            # No same half-axis placements
            if (new_obj.pos[0] == 0 and obj.pos[0] == 0 and 
                new_obj.pos[1] * obj.pos[1] > 0):
                return False
            if (new_obj.pos[1] == 0 and obj.pos[1] == 0 and 
                new_obj.pos[0] * obj.pos[0] > 0):
                return False
            
            # Check quadrant-specific constraints
            if _in_same_quadrant(new_obj, obj):
                # For quadrants I and III, use > 0 condition
                # For quadrants II and IV, use < 0 condition
                sign_check = 0
                if (new_obj.pos[0] * new_obj.pos[1] > 0):  # Quadrants I or III
                    sign_check = (new_obj.pos[0] - obj.pos[0]) * (new_obj.pos[1] - obj.pos[1])
                    if sign_check > 0:
                        return False
                else:  # Quadrants II or IV
                    sign_check = (new_obj.pos[0] - obj.pos[0]) * (new_obj.pos[1] - obj.pos[1])
                    if sign_check < 0:
                        return False
                        
        return True
    
    def _in_same_quadrant(obj1: Object, obj2: Object) -> bool:
        """Check if two objects are in the same quadrant"""
        return (np.sign(obj1.pos[0]) == np.sign(obj2.pos[0]) and 
                np.sign(obj1.pos[1]) == np.sign(obj2.pos[1]) and
                not (obj1.pos[0] == 0 or obj1.pos[1] == 0 or 
                     obj2.pos[0] == 0 or obj2.pos[1] == 0))
    
    # Start with agent at origin
    objects = []
    
    # Select random objects
    random_indices = random_generator.choice(len(candidate_list), n, replace=False)
    names = [candidate_list[i] for i in random_indices]
    
    # Add objects that satisfy constraints
    obj_count = 0
    while obj_count < n:
        pos = random_generator.integers(room_range[0], room_range[1], (1, 2))[0]
        obj = Object(name=names[obj_count], pos=pos)
        
        if _is_valid_placement(objects[1:], obj):
            objects.append(obj)
            obj_count += 1
            
    return objects



def generate_allo2ego_objects(
    n: int,
    random_generator: np.random.Generator,
    room_range: list[int] = [-10, 10],
    candidate_list: list[str] = CANDIDATE_OBJECTS,
) -> list[Object]:
    """Generate objects for allocentric-to-egocentric evaluation.
    
    Places objects in a sequence where an agent can visit each one
    by moving only in cardinal directions (0°, 90°, 180°, 270°).
    
    Args:
        n: Number of objects to generate
        random_generator: NumPy random generator
        room_range: Min/max room coordinates
        candidate_list: Possible object types
        
    Returns:
        List of positioned objects
    """
    # Calculate step size range
    min_step = 1
    max_step = (room_range[1] - room_range[0]) // 4
    
    # Select random objects
    indices = random_generator.choice(len(candidate_list), n, replace=False)
    names = [candidate_list[i] for i in indices]
    
    # Direction vectors (up, right, down, left)
    directions = [
        np.array([0, 1]),   # up
        np.array([1, 0]),   # right
        np.array([0, -1]),  # down
        np.array([-1, 0])   # left
    ]
    
    # Initialize with first object at origin
    objects = [Object(name=names[0], pos=np.array([0, 0]))]
    
    # Generate path of objects
    for i in range(1, n):
        current_pos = objects[-1].pos
        
        while True:
            # Choose random direction and step size
            dir_idx = random_generator.integers(0, 4)
            step_size = random_generator.integers(min_step, max_step + 1)
            
            # Calculate new position
            direction_vector = directions[dir_idx] * step_size
            new_pos = current_pos + direction_vector
            
            # Check if position is free
            if not any(np.array_equal(obj.pos, new_pos) for obj in objects):
                break
        
        # Add new object
        objects.append(Object(name=names[i], pos=new_pos))
    
    return objects



if __name__ == '__main__':
    room = generate_room_for_rotation_eval(
        n=10,
        random_generator=np.random.RandomState(42),
    )
    print(room)