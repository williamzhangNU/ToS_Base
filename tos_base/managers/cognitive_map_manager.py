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

from ..core.room import Room, BaseRoom
from ..core.object import Object, Agent
from ..core.relationship import TotalRelationship

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
- Always first output cognitive map before reasoning about other tasks in your thinking; never reverse the order.

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

**Task**: Maintain a global 2D cognitive map ({grid_size}×{grid_size}) of the room.

### Setup
- Origin [0,0] = initial position
- +Y axis = initial facing direction
- Track all objects including agent/yourself (position & orientation)

### Mapping Rules
- Convert relative positions to coordinates (e.g., front-left → x<0, y>0)
- Convert egocentric directions to global (e.g., forward → north)
- Assign confidence: high (certain), medium (estimated), low (unknown)
- Update map with new observations
- Default: unobserved objects at [0,0], orientation unknown, confidence low
- You won’t always be at the origin: every move or turn updates your position or orientation.

### Output Rules
- Always first output cognitive map before reasoning about other tasks in your thinking; never reverse the order.

### JSON OUTPUT FORMAT:
MUST follow this format to output cognitive map:
```json
{{
  "agent": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
  "object_name_1": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
}}
```
"""

COGMAP_EXP_REQUIRED_INSTRUCTION = """
In your thinking (in <think> ... </think>), you MUST follow the following steps:
Step 1: Briefly reason about cognitive map
Step 2: Output it strictly following:
```json
{{
  "agent": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
  "object_name_1": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
}}
```
Step 3: Reason about exploration.
Then provide only the answer in <answer> ... </answer>
"""

COGMAP_EVAL_REQUIRED_INSTRUCTION = """
In your thinking (in <think> ... </think>), you MUST follow the following steps:
Step 1: Briefly reason about cognitive map
Step 2: Output it strictly following:
```json
{{
  "agent": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
  "object_name_1": {{"position": [x, y], "facing": "direction", "confidence": "high/medium/low"}},
}}
```
Step 3: Reason about the question.
Then provide only the answer in <answer> ... </answer>
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
    pred_agent_state: Optional['Agent'] = None

    def to_dict(self):
        return {
            "dir_sim": self.dir_sim,
            "facing_sim": self.facing_sim,
            "pos_sim": self.pos_sim,
            "overall_sim": self.overall_sim,
            "extraction_success": self.extraction_success,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {},
            "pred_agent_state": self.pred_agent_state.to_dict() if self.pred_agent_state else {}
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
        
    def evaluate_cognitive_map(self, assistant_response: str, gt_room: Room, gt_agent: Agent) -> Optional[Dict[str, float]]:
        """
        Evaluate cognitive map from assistant response against ground truth.
        
        Args:
            assistant_response: Assistant response containing JSON cognitive map
            
        Returns:
            Dictionary with similarity metrics or None if extraction fails
        """
        # Extract cognitive map from response
        extracted = self._extract_json_and_create_room(assistant_response)
        
        if extracted is None or gt_room is None:
            # Log failed extraction
            self.turn_logs.append(CognitiveMapTurnLog())
            return None
        pred_room, pred_agent = extracted
        # Compare with ground truth using provided agent if any
        metrics = self._compare_rooms((pred_room, pred_agent), (gt_room.copy(), gt_agent))
        
        # Log successful evaluation
        turn_log = CognitiveMapTurnLog(**metrics, extraction_success=True, pred_room_state=pred_room, pred_agent_state=pred_agent)
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

    def _extract_json_and_create_room(self, model_response: str, room_name: str = "extracted_room") -> Optional[tuple[BaseRoom, Agent]]:
        """Extract JSON from model response and create (Room, Agent)."""
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
    
    def _create_room_from_json(self, json_data: Dict[str, Any], room_name: str = "extracted_room") -> Optional[tuple[BaseRoom, Agent]]:
        """Create (Room, Agent) from JSON data."""
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

            room = BaseRoom(objects=objects, name=room_name)
            return room, agent
            
        except Exception as e:
            print(f"Error creating Room object: {e}")
            return None

    @staticmethod
    def _transform_object_using_anchor(anchor: Object, target: Object) -> Object:
        """Set origin to anchor.pos and +Y to anchor.ori; return transformed copy of target."""
        target = target.copy()
        ori_to_R = {
            (0, 1): np.array([[1, 0], [0, 1]]),
            (1, 0): np.array([[0, 1], [-1, 0]]),
            (0, -1): np.array([[-1, 0], [0, -1]]),
            (-1, 0): np.array([[0, -1], [1, 0]]),
        }
        R = ori_to_R.get(tuple(anchor.ori.tolist()), ori_to_R[(0, 1)])
        p = (R @ (target.pos.astype(float) - anchor.pos.astype(float))).astype(float)
        o = target.ori
        if target.has_orientation:
            o = (R @ target.ori.astype(float)).astype(int)
        target.pos, target.ori = p, o
        return target
    
    def _compare_rooms(self, pred: tuple[BaseRoom, Agent], gt: tuple[BaseRoom, Agent]) -> Dict[str, float]:
        """Compare predicted and ground truth (room, agent)."""
        anchor = Object(name="anchor", pos=gt[1].init_pos, ori=gt[1].init_ori)
        pred_object_lists = [self._transform_object_using_anchor(anchor, obj) for obj in (pred[0].objects + [pred[1]])]
        gt_object_lists = [self._transform_object_using_anchor(anchor, obj) for obj in (gt[0].objects + [gt[1]])]

        dir_sim = self._calculate_dir_sim(pred_object_lists, gt_object_lists)
        facing_sim = self._calculate_facing_sim(pred_object_lists, gt_object_lists)
        pos_sim = self._calculate_pos_sim(pred_object_lists, gt_object_lists)
        
        overall_sim = 0.5 * dir_sim + 0.2 * facing_sim + 0.3 * pos_sim
        
        return {
            "dir_sim": dir_sim,
            "facing_sim": facing_sim,
            "pos_sim": pos_sim,
            "overall_sim": overall_sim
        }
    
    def _calculate_dir_sim(self, pred_object_lists: List[Object], gt_object_lists: List[Object]) -> float:
        """
        Compute similarity between predicted and ground truth room based on directional relationships.
        """
        pred = {o.name: o for o in pred_object_lists}
        tot = cor = 0.0
        for i in range(len(gt_object_lists)):
            for j in range(i + 1, len(gt_object_lists)):
                a, b = gt_object_lists[i], gt_object_lists[j]
                gt_rel = TotalRelationship.get_direction(a.pos, b.pos)
                p1, p2 = pred.get(a.name), pred.get(b.name)
                if p1 and p2:
                    pr = TotalRelationship.get_direction(p1.pos, p2.pos)
                    if (pr.horiz, pr.vert) == (gt_rel.horiz, gt_rel.vert): cor += 1.0
                tot += 1.0
        return cor / tot if tot else 0.0

    def _calculate_facing_sim(self, pred_object_lists: List[Object], gt_object_lists: List[Object]) -> float:
        pred = {o.name: o for o in pred_object_lists}
        tot = cor = 0.0
        for g in gt_object_lists:
            if not getattr(g, "has_orientation", False): 
                continue
            p = pred.get(g.name)
            if p is None: 
                return 0.0
            tot += 1.0
            if np.array_equal(p.ori, g.ori): 
                cor += 1.0
        return cor / tot if tot else 0.0


    
    def _calculate_pos_sim(self, pred_object_lists: List[Object], gt_object_lists: List[Object]) -> float:
        """
        Compute similarity between predicted and ground truth room.

        Formulas:
        s*  = (∑_i r_i·e_i) / (∑_i e_i·e_i)
        RMSE = (1/N) ∑_i ‖(s*)·e_i – r_i‖²
        L_r  = (1/N) ∑_i ‖r_i‖²
        ERR  = RMSE / L_r
        similarity = exp(−rmse/L)

        Args:
            pred_object_lists (List[Object]): predicted object lists
            gt_object_lists (List[Object]): ground truth object lists

        Returns:
            similarity (float): similarity between predicted and ground truth room.
        """
        pred = {o.name: o for o in pred_object_lists}
        gt_names = [o.name for o in gt_object_lists]
        if any(n not in pred for n in gt_names):  # TODO deal with incomplete objects
            return 0.0

        P1 = np.array([pred[n].pos for n in gt_names])
        P2 = np.array([o.pos for o in gt_object_lists])

        # scale-only alignment s = argmin ||s·P1 − P2||
        den = float((P1 * P1).sum())
        if den == 0.0: 
            return 0.0
        scale = float((P2 * P1).sum()) / den
        rmse = np.sqrt(((P1 * scale - P2) ** 2).sum(axis=1).mean())
        L = np.sqrt((P2 ** 2).sum(axis=1).mean())
        return float(np.exp(-rmse / L)) if L > 0 else 0.0
    


if __name__ == "__main__":
    pass