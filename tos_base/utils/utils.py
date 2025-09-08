import re
from typing import Tuple
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from .. import (
    EvaluationTurnLog,
    Room,
    ExplorationTurnLog,
    CognitiveMapTurnLog,
    Agent,
)
@dataclass
class EnvTurnLog:
    """Log data for a single environment turn."""
    turn_number: int
    user_message: str = ""  # Environment observation
    assistant_raw_message: str = ""  # Raw assistant input
    assistant_think_message: str = ""  # Think part of assistant message
    assistant_parsed_message: str = ""  # Parsed assistant action
    is_exploration_phase: bool = False
    exploration_log: Optional["ExplorationTurnLog"] = None
    evaluation_log: Optional["EvaluationTurnLog"] = None
    cogmap_log: Optional["CognitiveMapTurnLog"] = None
    cogmap_final_log: Optional["CognitiveMapTurnLog"] = None
    room_state: Optional["Room"] = None
    agent_state: Optional["Agent"] = None
    room_image: Optional[str] = None
    observed_items: List[str] = field(default_factory=list)
    cognitive_map_response: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "assistant_raw_message": self.assistant_raw_message,
            "assistant_think_message": self.assistant_think_message,
            "assistant_parsed_message": self.assistant_parsed_message,
            "is_exploration_phase": self.is_exploration_phase,
            "exploration_log": self.exploration_log.to_dict() if self.exploration_log else {},
            "evaluation_log": self.evaluation_log.to_dict() if self.evaluation_log else {},
            "cogmap_log": self.cogmap_log.to_dict() if self.cogmap_log else {},
            "cogmap_final_log": self.cogmap_final_log.to_dict() if self.cogmap_final_log else {},
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "observed_items": self.observed_items,
            "cognitive_map_response": self.cognitive_map_response,
            "room_image": self.room_image,
            "info": self.info
        }

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """Extract think and answer content from text using regex patterns"""
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else text
    
    return think_content, answer_content