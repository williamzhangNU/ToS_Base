from dataclasses import dataclass
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

    def __init__(self, config, room, agent,  dir = ".cache"):
        self.responses = []
        self.images = []
        self.current_turn = 0
        self.dir = os.path.abspath(os.path.join(dir, self.generate_unique_name(config, room, agent)))
        self.path = os.path.join(self.dir, "env_history.json")

        override = config.kwargs.get('override', False)
        if override:
            if os.path.exists(self.dir):
                shutil.rmtree(self.dir) 
        if self.is_history_exist():
            self.load()
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
        
    
    def is_history_exist(self):
        return os.path.exists(self.path)
    
    def generate_unique_name(self, config, room, agent):
        config_dict = config.get_observation_config()
        model_dict = config.kwargs['model_config']
        room_dict = room.to_dict()
        agent_dict = agent.to_dict()
        config_dict.update(model_dict)
        config_dict.update(room_dict)
        config_dict.update(agent_dict)
        config_str = json.dumps(config_dict, sort_keys=True)
        
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16] 
    
    def load(self) -> None:
        with open(self.path, "r") as f:
            data = json.load(f)
            self.responses = data['responses']
            self.images = data['images']    

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
    
    def update_response(self, response: Union[str, Dict[str, Any]], room_state, agent_state):
        self.responses.append(response)
        img_path = os.path.join(self.dir, "images", f"room_turn_{len(self.responses)}.png")
        RoomPlotter.plot(room_state, agent_state, mode='img', save_path=img_path)
        self.images.append(img_path)
        return img_path

    def get_responses(self) -> List[Union[str, Dict[str, Any]]]:
        return self.responses
