"""
Cognitive Map Manager

Handles cognitive map creation, validation, and comparison functionality.
Follows the same pattern as ExplorationManager and EvaluationManager.
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import copy

from ..core.room import Room
from ..core.object import Object, Agent
from ..utils.room_utils import set_initial_pos_as_origin

COGMAP_INSTRUCTION = """\
## Cognitive Map Creation

**Objective**  
Maintain a global 2D cognitive map of the room.

### Coordinate System:
- Use a {grid_size}x{grid_size} grid
- Your initial position as origin `[0, 0]`
- Your initial facing direction as positive Y-axis.
- Include all objects, including agent, i.e. your current position and orientation.

### Update Notes
- Record position as 2D coordinate and record orientation if given (north/south/east/west).
- Confidence: Set "confidence" to "high" (very sure), "medium" (uncertain or estimated), or "low" (unknown).
- Update current global map using local observations, note current global map may be incomplete or inaccurate.  
- Assign coordinates based on their spatial relationships
- Output full map: list **ALL** objects with `[x, y]` coordinates and orientation. For unobserved objects, use `[0, 0]` for position, `unknown` for orientation, and `low` for confidence.

### Rules
- An object can be anywhere at {grid_size}x{grid_size} grid.
- A at your right-front, it can be anywhere at your first quadrant, e.g., [1, 3], [4, 2], ...

### REQUIRED JSON OUTPUT FORMAT:
You MUST include this exact JSON structure in your thinking:
```json
{{
  "object_name_1": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
}}
```

### Example (MUST follow this format):
If a table is front right of you facing north:
```json
{{
  "table": {{"position": [3, 2], "facing": "north", "confidence": "medium"}},
}}
```
"""

COGMAP_INSTRUCTION_SHORTER = """\
## Cognitive Map Creation

**Objective**  
Maintain a global 2D cognitive map of the room.

- Grid: `{grid_size}×{grid_size}`, origin `[0,0]` is your initial position, facing +Y is your initial facing direction.
- Example Input: chair is at front-left, estimate the object's `[x,y]` anywhere in that sector (e.g. front-left ⇒ x<0, y>0).
- Record orientation if mentioned.
- Assign a "confidence" of high (certain), medium (estimated), or low (unknown).
- Merge into and update the global map; unobserved objects stay at `[0,0]`, orientation `unknown`, confidence `low`. 

### JSON OUTPUT FORMAT:
You MUST include this exact JSON structure in your thinking:
```json
{{
  "object_name_1": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
}}
```
"""

COGMAP_REQUIRED_INSTRUCTION = """
You MUST always output a json cognitive map in your thinking section, strictly follow the format below:
```json
{{
  "object_name_1": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
}}
```
"""

@dataclass
class CognitiveMapTurnLog:
    """Log data for a single cognitive map evaluation turn."""
    dir_sim: float = 0.0
    facing_sim: float = 0.0
    pos_sim: float = 0.0
    overall_sim: float = 0.0
    extraction_success: bool = False
    pred_room_state: Optional['Room'] = None

    def to_dict(self):
        return {
            "dir_sim": self.dir_sim,
            "facing_sim": self.facing_sim,
            "pos_sim": self.pos_sim,
            "overall_sim": self.overall_sim,
            "extraction_success": self.extraction_success,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {}
        }


class CognitiveMapManager:
    """
    Manages cognitive map creation, validation, and comparison.
    Follows the same pattern as ExplorationManager and EvaluationManager.
    """
    
    DEFAULT_COGMAP_SUMMARY = {
        "avg_dir_sim": 0.0,
        "avg_facing_sim": 0.0,
        "avg_pos_sim": 0.0,
        "avg_overall_sim": 0.0,
        "extraction_success_rate": 0.0,
        'n_successful': 0,
        "n_evaluations": 0,
    }
    
    def __init__(self, cogmap_type: str = "standard", grid_size: int = 5):
        """Initialize cognitive map manager with ground truth room."""
        self.turn_logs: List[CognitiveMapTurnLog] = []
        self.cogmap_summary = copy.deepcopy(self.DEFAULT_COGMAP_SUMMARY)

        self.config = {
            "cogmap_type": cogmap_type,
            "grid_size": grid_size
        }

    def get_cognitive_map_instruction(self) -> str:
        assert self.config['cogmap_type'] == "standard", "Only standard format is supported"
        return COGMAP_INSTRUCTION_SHORTER.format(grid_size=self.config["grid_size"])
        
    def evaluate_cognitive_map(self, assistant_response: str, gt_room: Room) -> Optional[Dict[str, float]]:
        """
        Evaluate cognitive map from assistant response against ground truth.
        
        Args:
            assistant_response: Assistant response containing JSON cognitive map
            
        Returns:
            Dictionary with similarity metrics or None if extraction fails
        """
        # Extract cognitive map from response
        extracted_room = self._extract_json_and_create_room(assistant_response)
        
        if extracted_room is None or gt_room is None:
            # Log failed extraction
            self.turn_logs.append(CognitiveMapTurnLog())
            return None
        
        # Compare with ground truth
        metrics = self._compare_rooms(extracted_room, gt_room.copy())
        
        # Log successful evaluation
        turn_log = CognitiveMapTurnLog(**metrics, extraction_success=True, pred_room_state=extracted_room)
        self.turn_logs.append(turn_log)
        
        return metrics
            
    
    def get_cogmap_summary(self) -> Dict[str, Any]:
        """Get cognitive map summary statistics."""
        if not self.turn_logs:
            return {**self.DEFAULT_COGMAP_SUMMARY, "n_evaluations": 0}
        
        # Calculate averages from turn logs
        successful_logs = [log for log in self.turn_logs if log.extraction_success]
        n_evaluations = len(self.turn_logs)
        n_successful = len(successful_logs)
        
        if n_successful == 0:
            return {
                **self.DEFAULT_COGMAP_SUMMARY,
                "n_evaluations": n_evaluations,
                "extraction_success_rate": 0.0
            }
        
        avg_directional = sum(log.dir_sim for log in successful_logs) / n_successful
        avg_facing = sum(log.facing_sim for log in successful_logs) / n_successful  
        avg_position = sum(log.pos_sim for log in successful_logs) / n_successful
        avg_overall = sum(log.overall_sim for log in successful_logs) / n_successful
        success_rate = n_successful / n_evaluations
        
        return {
            "avg_dir_sim": avg_directional,
            "avg_facing_sim": avg_facing,
            "avg_pos_sim": avg_position,
            "avg_overall_sim": avg_overall,
            "extraction_success_rate": success_rate,
            "n_successful": n_successful,
            "n_evaluations": n_evaluations
        }
    
    @staticmethod
    def aggregate_group_performance(cogmap_summaries: List[Dict]) -> Dict[str, float]:
        """Calculate cognitive map performance for a group."""
        if not cogmap_summaries:
            return {
                "overall_avg_dir_sim": 0.0,
                "overall_avg_facing_sim": 0.0, 
                "overall_avg_pos_sim": 0.0,
                "overall_avg_overall_sim": 0.0,
                "overall_avg_extraction_success_rate": 0.0,
                "overall_avg_evaluations": 0.0,
                "overall_n_successful": 0
            }
        
        return {
            "overall_avg_dir_sim": sum(s.get('avg_dir_sim', 0) for s in cogmap_summaries) / len(cogmap_summaries),
            "overall_avg_facing_sim": sum(s.get('avg_facing_sim', 0) for s in cogmap_summaries) / len(cogmap_summaries),
            "overall_avg_pos_sim": sum(s.get('avg_pos_sim', 0) for s in cogmap_summaries) / len(cogmap_summaries),
            "overall_avg_overall_sim": sum(s.get('avg_overall_sim', 0) for s in cogmap_summaries) / len(cogmap_summaries),
            "overall_avg_extraction_success_rate": sum(s.get('extraction_success_rate', 0) for s in cogmap_summaries) / len(cogmap_summaries),
            "overall_avg_evaluations": sum(s.get('n_evaluations', 0) for s in cogmap_summaries) / len(cogmap_summaries),
            "overall_n_successful": sum(s.get('n_successful', 0) for s in cogmap_summaries)
        }
    



    # =============================== Helper Functions =============================== 

    def _extract_json_and_create_room(self, model_response: str, room_name: str = "extracted_room") -> Optional[Room]:
        """Extract JSON from model response and create Room object."""
        json_data = self._extract_json_from_text(model_response)
        if json_data is None:
            return None
        return self._create_room_from_json(json_data, room_name)
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON content from text."""
        patterns = [
            r'```json\s*\n(.*?)\n\s*```',
            r'```\s*\n(.*?)\n\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    json_text = match.strip()
                    json_data = json.loads(json_text)
                    if isinstance(json_data, dict) and json_data:
                        return json_data
                except (json.JSONDecodeError, ValueError):
                    continue
        return None
    
    def _create_room_from_json(self, json_data: Dict[str, Any], room_name: str = "extracted_room") -> Optional[Room]:
        """Create Room object from JSON data."""
        try:
            objects = []
            direction_mapping = {
                "north": np.array([0, 1]),
                "south": np.array([0, -1]),
                "east": np.array([1, 0]),
                "west": np.array([-1, 0])
            }
            agent = Agent(name="agent")
            
            for obj_name, obj_info in json_data.items():
                
                if 'position' not in obj_info:
                    continue
                
                position = obj_info['position']
                if not isinstance(position, list) or len(position) != 2:
                    continue
                
                pos = np.array([float(position[0]), float(position[1])])
                
                facing = obj_info.get('facing', 'north')
                has_orientation = True
                if isinstance(facing, str):
                    facing = facing.lower()
                    ori = direction_mapping.get(facing, direction_mapping['north'])
                else:
                    has_orientation = False
                    ori = np.array([0, 0])
                
                if obj_name.lower() in ['agent', 'you', 'player']:
                    agent = Agent(name='agent', pos=pos, ori=ori, has_orientation=has_orientation)
                else:
                    obj = Object(name=obj_name, pos=pos, ori=ori, has_orientation=has_orientation)
                    objects.append(obj)
            
            if not objects:
                return None
            
            room = Room(agent=agent, objects=objects, name=room_name, initial_pos=Object(name='initial_pos', pos=np.array([0, 0]), ori=np.array([0, 1])))
            
            return room
            
        except Exception as e:
            print(f"Error creating Room object: {e}")
            return None
    
    def _compare_rooms(self, pred_room: Room, gt_room: Room) -> Dict[str, float]:
        """Compare cognitive map room with ground truth room."""
        norm_pred_room = set_initial_pos_as_origin(pred_room)
        norm_gt_room = set_initial_pos_as_origin(gt_room)

        dir_sim = self._calculate_dir_sim(norm_pred_room, norm_gt_room)
        facing_sim = self._calculate_facing_sim(norm_pred_room, norm_gt_room)
        pos_sim = self._calculate_pos_sim(norm_pred_room, norm_gt_room)
        
        overall_sim = 0.5 * dir_sim + 0.2 * facing_sim + 0.3 * pos_sim
        
        return {
            "dir_sim": dir_sim,
            "facing_sim": facing_sim,
            "pos_sim": pos_sim,
            "overall_sim": overall_sim
        }
    
    def _calculate_dir_sim(self, pred_room: Room, gt_room: Room) -> float:
        """
        Compute similarity between predicted and ground truth room based on directional relationships.
        """
        obj_name_list = [obj.name for obj in gt_room.valid_objects]
        total_pairs, correct_pairs = 0.0, 0.0
        for i in range(len(obj_name_list)):
            for j in range(i + 1, len(obj_name_list)):
                obj1_name = obj_name_list[i]
                obj2_name = obj_name_list[j]
                dir_pair_gt, _ = gt_room.get_direction(obj1_name, obj2_name, perspective='allo')

                try:
                    dir_pair_pred, _ = pred_room.get_direction(obj1_name, obj2_name, perspective='allo')
                    if dir_pair_pred.horiz == dir_pair_gt.horiz and dir_pair_pred.vert == dir_pair_gt.vert:
                        correct_pairs += 1.0
                except Exception as e:
                    print(f"Error calculating direction similarity: {e}")
                    continue
                total_pairs += 1.0
        
        return correct_pairs / total_pairs if total_pairs > 0 else 0.0

    def _calculate_facing_sim(self, pred_room: Room, gt_room: Room) -> float:
        """
        Compute similarity between predicted and ground truth room based on orientation consistency.
        """
        obj_name_list = [obj.name for obj in gt_room.valid_objects]
        total_ori, correct_ori = float(len(obj_name_list)), 0.0
        
        for obj_name in obj_name_list:
            obj2 = gt_room.get_object_by_name(obj_name)
            try:
                obj1 = pred_room.get_object_by_name(obj_name)
                if obj2.has_orientation:
                    if np.array_equal(obj1.ori, obj2.ori):
                        correct_ori += 1.0
            except Exception as e:
                print(f"Error calculating facing similarity: {e}")
                continue
        
        return correct_ori / total_ori if total_ori > 0 else 0.0


    
    def _calculate_pos_sim(self, pred_room: Room, gt_room: Room) -> float:
        """
        Compute similarity between predicted and ground truth room.

        Formulas:
        s*  = (∑_i r_i·e_i) / (∑_i e_i·e_i)
        RMSE = (1/N) ∑_i ‖(s*)·e_i – r_i‖²
        L_r  = (1/N) ∑_i ‖r_i‖²
        ERR  = RMSE / L_r
        similarity = exp(−rmse/L)

        Args:
            pred_room (Room): predicted room
            gt_room (Room): ground truth room
            common_obj_names (set): common object names

        Returns:
            similarity (float): similarity between predicted and ground truth room.
        """
        obj_name_list = [obj.name for obj in gt_room.valid_objects]
        pred_obj_name_list = [obj.name for obj in pred_room.valid_objects]
        if set(obj_name_list) != set(pred_obj_name_list): # TODO deal with incomplete objects
            return 0.0
        
        P1 = np.array([pred_room.get_object_by_name(n).pos for n in pred_obj_name_list])
        P2 = np.array([gt_room.get_object_by_name(n).pos for n in obj_name_list])

        # scale-only alignment s = argmin ||s·P1 − P2||
        num = (P2 * P1).sum()
        den = (P1 * P1).sum()
        scale = num / den if den > 0 else 0.0
        P1_scaled = P1 * scale

        # RMSE between scaled estimates and ground truth
        rmse = np.sqrt(((P1_scaled - P2)**2).sum(axis=1).mean())

        # normalization length L = RMS distance of all objects from agent (origin)
        L = np.sqrt((P2**2).sum(axis=1).mean())
        
        # similarity = exp(−rmse/L)
        return float(np.exp(-rmse / L)) if L > 0 else 0.0
    


if __name__ == "__main__":
    room = Room(
        agent=Agent(name="agent", pos=np.array([0, 0]), ori=np.array([0, 1])),
        objects=[Object(name="table", pos=np.array([1, 1]), ori=np.array([0, 1])), Object(name="chair", pos=np.array([0, 1]), ori=np.array([0, 1]))],
        name="room"
    )
    room2 = Room(
        agent=Agent(name="agent", pos=np.array([0, 0]), ori=np.array([0, 1])),
        objects=[Object(name="table", pos=np.array([-1,-1]), ori=np.array([0, 1])), Object(name="chair", pos=np.array([0, 2]), ori=np.array([0, 1]))],
        name="room2"
    )

    common_obj_names = set(["table", "chair"])
    print(CognitiveMapManager()._calculate_pos_sim(room, room2, common_obj_names))