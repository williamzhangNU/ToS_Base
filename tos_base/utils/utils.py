"""
JSON Extractor Module

This module provides functionality to extract JSON-formatted room layout information
from model responses and create Room objects.

Main functions:
1. extract_json_and_create_room: Extract JSON from model response and create Room object
2. extract_json_from_text: Extract JSON content from text
3. create_room_from_json: Create Room object from JSON data
4. validate_room_json: Validate JSON data format

Supported JSON format example:
```json
{
  "whiteboard": {"position": [1, 1], "facing": "east"},
  "oven": {"position": [0, 1], "facing": "east"},
  "chair": {"position": [-1, 1], "facing": "east"}
}
```

Supported orientations: north, south, east, west
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add path to support relative imports
try:
    from ..core.room import Room
    from ..core.object import Object, Agent
except ImportError:
    # If relative import fails, try absolute import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.room import Room
    from core.object import Object, Agent


def extract_json_and_create_room(model_response: str, room_name: str = "extracted_room", use_improved_plot: bool = True) -> Optional[Room]:
    """
    Extract JSON-formatted room layout information from model response and create Room object

    Args:
        model_response: Model response text containing JSON-formatted room layout
        room_name: Room name, default is "extracted_room"
        use_improved_plot: Whether to use improved visualization function, default is True

    Returns:
        Room object, or None if extraction fails

    Example:
        >>> response = '''
        ... Based on the description, the room layout is as follows:
        ... ```json
        ... {
        ...   "whiteboard": {"position": [1, 1], "facing": "east"},
        ...   "oven": {"position": [0, 1], "facing": "east"},
        ...   "chair": {"position": [-1, 1], "facing": "east"}
        ... }
        ... ```
        ... '''
        >>> room = extract_json_and_create_room(response)
        >>> room.plot_improved()  # Use improved visualization
    """

    # Extract JSON content
    json_data = extract_json_from_text(model_response)
    if json_data is None:
        return None

    # Create Room object
    room = create_room_from_json(json_data, room_name)

    # Add improved visualization method to Room object
    if room and use_improved_plot:
        room.plot_improved = lambda: plot_room_with_orientations(room)

    return room


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from text

    Args:
        text: Text containing JSON

    Returns:
        Parsed JSON dictionary, or None if extraction fails
    """

    # Try multiple JSON extraction patterns
    patterns = [
        # Standard ```json code block
        r'```json\s*\n(.*?)\n\s*```',
        # Code block without language identifier
        r'```\s*\n(.*?)\n\s*```',
        # Direct JSON object (starting with { and ending with })
        r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Clean the matched text
                json_text = match.strip()
                # Try to parse JSON
                json_data = json.loads(json_text)
                if isinstance(json_data, dict) and json_data:
                    return json_data
            except (json.JSONDecodeError, ValueError):
                continue

    return None


def create_room_from_json(json_data: Dict[str, Any], room_name: str = "extracted_room") -> Optional[Room]:
    """
    Create Room object from JSON data

    Args:
        json_data: JSON dictionary containing room layout information
        room_name: Room name

    Returns:
        Room object, or None if creation fails
    """

    try:
        objects = []

        # Direction mapping: string to numpy array
        direction_mapping = {
            "north": np.array([0, 1]),
            "south": np.array([0, -1]),
            "east": np.array([1, 0]),
            "west": np.array([-1, 0])
        }

        # Iterate through each object in JSON
        for obj_name, obj_info in json_data.items():
            # Skip agent-related objects
            if obj_name.lower() in ['agent', 'you', 'player']:
                continue

            # Get position information
            if 'position' not in obj_info:
                print(f"Warning: Object {obj_name} missing position information")
                continue

            position = obj_info['position']
            if not isinstance(position, list) or len(position) != 2:
                print(f"Warning: Object {obj_name} has incorrect position format: {position}")
                continue

            pos = np.array([float(position[0]), float(position[1])])

            # Get orientation information
            facing = obj_info.get('facing', 'north')
            if isinstance(facing, str):
                facing = facing.lower()
                if facing in direction_mapping:
                    ori = direction_mapping[facing]
                else:
                    print(f"Warning: Object {obj_name} orientation '{facing}' not recognized, using default 'north'")
                    ori = direction_mapping['north']
            else:
                print(f"Warning: Object {obj_name} has incorrect orientation format: {facing}, using default 'north'")
                ori = direction_mapping['north']

            # Create object
            obj = Object(name=obj_name, pos=pos, ori=ori, has_orientation=True)
            objects.append(obj)

        if not objects:
            print("Warning: No valid objects found")
            return None

        # Create default agent
        agent = Agent(name="agent", pos=np.array([0, 0]), ori=np.array([0, 1]))

        # Create Room object
        room = Room(agent=agent, objects=objects, name=room_name)

        return room

    except Exception as e:
        print(f"Error occurred while creating Room object: {e}")
        return None


def validate_room_json(json_data: Dict[str, Any]) -> bool:
    """
    Validate whether JSON data meets room layout format requirements

    Args:
        json_data: JSON dictionary to validate

    Returns:
        True if format is correct, False otherwise
    """

    if not isinstance(json_data, dict):
        return False

    valid_directions = {'north', 'south', 'east', 'west'}

    for _, obj_info in json_data.items():
        if not isinstance(obj_info, dict):
            return False

        # Check position information
        if 'position' not in obj_info:
            return False

        position = obj_info['position']
        if not isinstance(position, list) or len(position) != 2:
            return False

        try:
            float(position[0])
            float(position[1])
        except (ValueError, TypeError):
            return False

        # Check orientation information (optional)
        if 'facing' in obj_info:
            facing = obj_info['facing']
            if not isinstance(facing, str) or facing.lower() not in valid_directions:
                return False

    return True


def plot_room_with_orientations(room):
    """
    Improved room visualization function that better displays object orientations
    """
    min_x, max_x, min_y, max_y = room.get_boundary()
    min_x, max_x, min_y, max_y = int(min_x)-1, int(max_x)+1, int(min_y)-1, int(max_y)+1

    width, height = max_x - min_x + 1, max_y - min_y + 1
    grid = [[' '] * width for _ in range(height)]

    ori_map = {(0,1):'^', (0,-1):'v', (1,0):'>', (-1,0):'<'}
    labels = []

    # First place all objects
    for i, obj in enumerate([room.agent] + room.objects):
        x, y = int(obj.pos[0]) - min_x, max_y - int(obj.pos[1])
        symbol = 'A' if obj == room.agent else str(i-1)
        labels.append(f"{symbol}:{obj.name}")

        if 0 <= y < height and 0 <= x < width:
            # Combine object symbol and orientation symbol for display
            ori_symbol = ori_map.get(tuple(obj.ori), '?')
            grid[y][x] = f"{symbol}{ori_symbol}"

    print(f"\n--- {room.name} (Improved) ---")
    print("Legend:", " | ".join(labels))
    print("Symbol format: [Object][Orientation] (^:North v:South >:East <:West)")
    for y in range(height):
        print(f"{max_y-y:3d} " + "".join(f"{cell:>3}" for cell in grid[y]))
    print("    " + "".join(f"{min_x+x:>3}" for x in range(width)))


def create_room_and_visualize(model_response: str, room_name: str = "extracted_room") -> Optional[Room]:
    """
    Convenience function: Extract JSON, create Room object, and immediately display improved visualization

    Args:
        model_response: Model response text containing JSON-formatted room layout
        room_name: Room name, default is "extracted_room"

    Returns:
        Room object, or None if extraction fails
    """
    room = extract_json_and_create_room(model_response, room_name)
    if room:
        print(f"Successfully created Room object: {room_name}")
        plot_room_with_orientations(room)
        return room
    else:
        print("Failed to create Room object")
        return None


def calculate_position_similarity(room1: Room, room2: Room, common_obj_names: set) -> float:
    """
    Calculate overall position similarity between objects in two rooms, allowing scaling but penalizing position deviations
    Simplified version: Calculate average scale factor, then directly compute error and convert to 0-1 score

    Args:
        room1: First room
        room2: Second room (usually ground truth)
        common_obj_names: Set of common object names

    Returns:
        float: Position similarity score (0.0-1.0)
    """
    if len(common_obj_names) < 2:
        return 0.0

    # Get positions of all common objects in both rooms
    positions1 = []
    positions2 = []

    for obj_name in common_obj_names:
        try:
            obj1 = room1.get_object_by_name(obj_name)
            obj2 = room2.get_object_by_name(obj_name)
            positions1.append(obj1.pos)
            positions2.append(obj2.pos)
        except Exception as e:
            print(f"Warning: Error getting position of object {obj_name}: {e}")
            continue

    if len(positions1) < 2:
        return 0.0

    positions1 = np.array(positions1)
    positions2 = np.array(positions2)

    # Since agents are all at (0,0), directly use absolute positions to calculate scale factor
    # Calculate average scale factor
    scale_factors = []
    for pos1, pos2 in zip(positions1, positions2):
        norm1 = np.linalg.norm(pos1)  # Distance to origin (0,0)
        norm2 = np.linalg.norm(pos2)  # Distance to origin (0,0)
        if norm1 > 1e-6 and norm2 > 1e-6:
            scale_factors.append(norm2 / norm1)

    if not scale_factors:
        # All objects are at origin, check if they completely overlap
        return 1.0 if np.allclose(positions1, positions2) else 0.0

    # Use average as overall scale factor
    average_scale = np.mean(scale_factors)

    # Adjust room1 according to average scale factor
    scaled_pos1 = positions1 * average_scale

    # Calculate position error after adjustment
    total_error = 0.0
    for pos1_scaled, pos2 in zip(scaled_pos1, positions2):
        error = np.linalg.norm(pos1_scaled - pos2)
        total_error += error

    # Calculate average error
    average_error = total_error / len(positions1)

    # Convert error to 0-1 similarity score
    # Use exponential decay function: similarity=1 when error=0, similarity approaches 0 as error increases
    max_reasonable_error = 2.0  # Adjustable parameter: when error reaches this value, similarity is about 0.37
    similarity = np.exp(-average_error / max_reasonable_error)

    return similarity


def compare_room_consistency_text(room1: Room, room2: Room):
    """
    Compare relative relationships and orientation consistency of objects in two rooms

    Args:
        room1: First room (usually predicted room)
        room2: Second room (usually ground truth room)

    Returns:
        dict: Dictionary containing similarity metrics
               - directional_similarity: 0.0-1.0, proportion of consistent pairwise object relationships
               - facing_similarity: 0.0-1.0, proportion of consistent object orientations
               - position_similarity: 0.0-1.0, overall position similarity allowing scaling
               - overall_similarity: 0.0-1.0, weighted combination of all metrics
    """

    # Get common object names in both rooms (including agent, but excluding initial_pos)
    room1_obj_names = {obj.name for obj in room1.objects}
    room2_obj_names = {obj.name for obj in room2.objects}
    common_obj_names = room1_obj_names.intersection(room2_obj_names)

    # Add agent to common object list (agent always exists in both rooms)
    common_obj_names.add('agent')

    if len(common_obj_names) < 2:
        print("Warning: Less than 2 common objects, cannot compare relative relationships")
        return {"directional_similarity": 0.0, "facing_similarity": 0.0, "position_similarity": 0.0, "overall_similarity": 0.0}

    # Calculate relative relationship consistency
    total_pairs = 0
    consistent_pairs = 0

    # Iterate through all object pairs
    common_obj_list = list(common_obj_names)
    for i in range(len(common_obj_list)):
        for j in range(i + 1, len(common_obj_list)):
            obj1_name = common_obj_list[i]
            obj2_name = common_obj_list[j]

            try:
                # Get relative relationships in both rooms
                dir_pair1, _ = room1.get_direction(obj1_name, obj2_name, perspective='allo')
                dir_pair2, _ = room2.get_direction(obj1_name, obj2_name, perspective='allo')

                # Compare if relative relationships are consistent
                if dir_pair1.horiz == dir_pair2.horiz and dir_pair1.vert == dir_pair2.vert:
                    consistent_pairs += 1

                total_pairs += 1

            except Exception as e:
                print(f"Warning: Error comparing relative relationship between objects {obj1_name} and {obj2_name}: {e}")
                continue

    # Calculate relative relationship consistency ratio
    relation_consistency = consistent_pairs / total_pairs if total_pairs > 0 else 0.0

    # Calculate orientation consistency (excluding agent)
    total_orientations = 0
    consistent_orientations = 0

    for obj_name in common_obj_names:
        # Skip agent orientation comparison
        if obj_name == 'agent':
            continue

        try:
            obj1 = room1.get_object_by_name(obj_name)
            obj2 = room2.get_object_by_name(obj_name)

            # Only compare objects with orientation
            if obj1.has_orientation and obj2.has_orientation:
                # Compare if orientation vectors are the same
                if np.array_equal(obj1.ori, obj2.ori):
                    consistent_orientations += 1

                total_orientations += 1

        except Exception as e:
            print(f"Warning: Error comparing orientation of object {obj_name}: {e}")
            continue

    # Calculate orientation consistency ratio
    orientation_consistency = consistent_orientations / total_orientations if total_orientations > 0 else 0.0

    # Calculate position similarity (newly added metric)
    position_similarity = calculate_position_similarity(room1, room2, common_obj_names)

    result = {}
    result["directional_similarity"] = relation_consistency
    result["facing_similarity"] = orientation_consistency
    result["position_similarity"] = position_similarity
    result["overall_similarity"] = 0.5 * relation_consistency + 0.2 * orientation_consistency + 0.3 * position_similarity
    return result


# Test functions
if __name__ == "__main__":
    print("\n=== Test Case 4: Room Consistency Comparison ===")

    # Create two similar rooms for comparison
    # Room A: Original room
    test_response_a = '''
    ```json
    {
      "table": {"position": [2, 1], "facing": "north"},
      "chair": {"position": [0, 1], "facing": "east"},
      "lamp": {"position": [-1, 2], "facing": "south"}
    }
    ```
    '''

    # Room B: Completely identical room
    test_response_b = '''
    ```json
    {
      "table": {"position": [2, 1], "facing": "north"},
      "chair": {"position": [0, 1], "facing": "east"},
      "lamp": {"position": [-1, 2], "facing": "south"}
    }
    ```
    '''

    # Room C: Partially different room (position changed)
    test_response_c = '''
    ```json
    {
      "table": {"position": [1, 2], "facing": "north"},
      "chair": {"position": [0, 1], "facing": "east"},
      "lamp": {"position": [-1, 2], "facing": "south"}
    }
    ```
    '''

    # Room D: Different orientation room
    test_response_d = '''
    ```json
    {
      "table": {"position": [4, 1], "facing": "south"},
      "chair": {"position": [0, 1], "facing": "west"},
      "lamp": {"position": [-1, 2], "facing": "north"}
    }
    ```
    '''

    room_a = extract_json_and_create_room(test_response_a, "room_a")
    room_b = extract_json_and_create_room(test_response_b, "room_b")
    room_c = extract_json_and_create_room(test_response_c, "room_c")
    room_d = extract_json_and_create_room(test_response_d, "room_d")
    plot_room_with_orientations(room_a)
    plot_room_with_orientations(room_b)
    plot_room_with_orientations(room_c)
    plot_room_with_orientations(room_d)
    if all([room_a, room_b, room_c, room_d]):
        print("Successfully created all test rooms")

        # Test completely identical rooms
        result = compare_room_consistency_text(room_a, room_b)
        print(f"\nRoom A vs Room B (Completely identical):")
        print(f"Directional consistency: {result['directional_similarity']:.2f}")
        print(f"Orientation consistency: {result['facing_similarity']:.2f}")
        print(f"Position consistency: {result['position_similarity']:.2f}")
        print(f"Overall similarity: {result['overall_similarity']:.2f}")

        # Test rooms with different positions
        result = compare_room_consistency_text(room_a, room_c)
        print(f"\nRoom A vs Room C (Partially different positions):")
        print(f"Directional consistency: {result['directional_similarity']:.2f}")
        print(f"Orientation consistency: {result['facing_similarity']:.2f}")
        print(f"Position consistency: {result['position_similarity']:.2f}")
        print(f"Overall similarity: {result['overall_similarity']:.2f}")

        # Test rooms with different orientations
        result = compare_room_consistency_text(room_a, room_d)
        print(f"\nRoom A vs Room D (Different orientations):")
        print(f"Directional consistency: {result['directional_similarity']:.2f}")
        print(f"Orientation consistency: {result['facing_similarity']:.2f}")
        print(f"Position consistency: {result['position_similarity']:.2f}")
        print(f"Overall similarity: {result['overall_similarity']:.2f}")

    else:
        print("Failed to create test rooms")
