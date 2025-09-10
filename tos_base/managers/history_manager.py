from typing import Optional, List, Dict, Any, Union
import os
import shutil
import json
import hashlib
from ..utils.room_utils import RoomPlotter

class HistoryManager:
    """Simple conversation history manager.
    save room images and responses of each exploration turn
    """

    def __init__(self, config, room, agent, override=False, override_cogmap=False, dir = ".cache"):
        self.responses = []
        self.images = []
        self.cogmap_responses = []  # Store cognitive map responses
        self.current_turn = 0
        self.dir = os.path.abspath(os.path.join(dir, self.generate_unique_name(config, room, agent)))
        self.path = os.path.join(self.dir, "env_history.json")
        self.cogmap_path = os.path.join(self.dir, "cogmap_history.json")

        if override:
            if os.path.exists(self.dir):
                shutil.rmtree(self.dir) 
        elif override_cogmap:
            if os.path.exists(self.cogmap_path):
                os.remove(self.cogmap_path)
        self.load()
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
        
    
    def is_history_exist(self):
        return os.path.exists(self.path)
    
    def generate_unique_name(self, config, room, agent):
        if isinstance(config, dict):
            config_dict = {**config['observation_config'], **config['model_config']}
        else:
            config_dict = {**config.get_observation_config(), **config.get_model_config()}

        if isinstance(room, dict):
            room_dict = room
        else:
            room_dict = room.to_dict()
        if isinstance(agent, dict):
            agent_dict = agent
        else: 
            agent_dict = agent.to_dict()

        config_dict.update(room_dict)
        config_dict.update(agent_dict)
        config_str = json.dumps(config_dict, sort_keys=True)
        
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16] 
    
    def load(self) -> None:
        if self.is_history_exist():
            with open(self.path, "r") as f:
                data = json.load(f)
                self.responses = data['responses']
                self.images = data['images']
        """Load cognitive map responses from separate file"""
        if os.path.exists(self.cogmap_path):
            with open(self.cogmap_path, "r") as f:
                data = json.load(f)
                self.cogmap_responses = data.get('cogmap_responses', [])    

    def get_image_path(self, turn_num):
        assert 0 < turn_num <= len(self.images)
        img_path = self.images[turn_num -1]
        return img_path
    
    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({
                "responses": self.responses,
                "images": self.images
            }, f, ensure_ascii=False, indent=2)
            
    def save_cogmap(self) -> None:
        """Save cognitive map responses to separate file"""
        if not os.path.exists(self.cogmap_path):
            with open(self.cogmap_path, "w") as f:
                json.dump({
                    "cogmap_responses": self.cogmap_responses
                }, f, ensure_ascii=False, indent=2)
    
    def update_response(self, response: Union[str, Dict[str, Any]], room_state, agent_state):
        self.responses.append(response)
        img_path = os.path.join(self.dir, "images", f"room_turn_{len(self.responses)}.png")
        RoomPlotter.plot(room_state, agent_state, mode='img', save_path=img_path)
        self.images.append(img_path)
        return img_path

    def get_responses(self) -> List[Union[str, Dict[str, Any]]]:
        return self.responses
    
    def get_cogmap_response(self, turn_idx: int) -> Optional[str]:
        """Get cognitive map response for a specific turn (0-indexed)"""
        if 0 <= turn_idx < len(self.cogmap_responses):
            return self.cogmap_responses[turn_idx]
        return None
    
    def update_cogmap_response(self, response: str):
        """Add a cognitive map response for the current turn"""
        self.cogmap_responses.append(response)
        
    def has_cogmap_response(self, turn_idx: int) -> bool:
        """Check if cognitive map response exists for a specific turn (0-indexed)"""
        return 0 <= turn_idx < len(self.cogmap_responses) and self.cogmap_responses[turn_idx] is not None
