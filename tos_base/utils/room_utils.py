import numpy as np
from typing import Dict, Any

from ..core.room import Room
from ..core.constant import CANDIDATE_OBJECTS, ObjectInfo
from ..core.object import Object, Agent

def generate_room(
        room_range: tuple[int, int],
        n_objects: int,
        generation_type: str,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: list[ObjectInfo] = CANDIDATE_OBJECTS,
    ) -> Room:
    """
    Generate a room based on the given configuration
    """

    if generation_type in ['rand', 'pov']:
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
        agent=Agent()
    )



def generate_random_objects(
    n: int,
    random_generator: np.random.Generator,
    room_range: list[int] = [-10, 10],
    perspective_taking: bool = False,
    candidate_list: list[ObjectInfo] = CANDIDATE_OBJECTS,
) -> list[Object]:
    """Generate random objects with random positions and orientations."""
    # Select random objects from candidate list
    indices = random_generator.choice(len(candidate_list), n, replace=False)
    selected_object_info = [candidate_list[i] for i in indices]
    
    # Generate random positions ensuring no two objects are at the same position
    positions = [np.array([0, 0])]
    while len(positions) < n + 1:
        pos = random_generator.integers(room_range[0], room_range[1], (1, 2))[0]
        if not any(np.array_equal(pos, existing_pos) for existing_pos in positions):
            positions.append(pos)
    positions = np.array(positions[1:])
    orientations = random_generator.integers(0, 4, n)
    
    # Map orientation values to vectors
    ori_vectors = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}
    
    # Create and return object list
    objects = []
    for obj_info, pos, ori_idx in zip(selected_object_info, positions, orientations):
        ori = np.array(ori_vectors[ori_idx]) if obj_info.has_orientation and perspective_taking else np.array([0, 1])
        objects.append(Object(name=obj_info.name, pos=pos, ori=ori, has_orientation=obj_info.has_orientation))

    return objects



def generate_room_for_rotation_eval(
    n: int,
    random_generator: np.random.Generator,
    room_range: list[int] = [-10, 10],
    candidate_list: list[ObjectInfo] = CANDIDATE_OBJECTS,
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
    selected_object_info = [candidate_list[i] for i in random_indices]
    
    # Add objects that satisfy constraints
    obj_count = 0
    while obj_count < n:
        pos = random_generator.integers(room_range[0], room_range[1], (1, 2))[0]
        
        obj_info = selected_object_info[obj_count]
        
        obj = Object(name=obj_info.name, pos=pos, has_orientation=obj_info.has_orientation)
        
        if _is_valid_placement(objects, obj):
            objects.append(obj)
            obj_count += 1
            
    return objects

if __name__ == '__main__':
    room = generate_room_for_rotation_eval(
        n=10,
        random_generator=np.random.default_rng(42),
    )
    print(room)