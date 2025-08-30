from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import os
import json
import hashlib

class HistoryManager:
    """Simple conversation history manager.

    - update(text, reward):
        If reward is not None, treats "text" as assistant message and stores reward.
        If reward is None, treats "text" as user message.
        do not need to store observation/user prompt, as it can be stepped from env again.
    """

    def __init__(self, seed, config , dir = ".cache"):
        self.responses = []
        self.initial_observation = None
        self.dir = dir
        # print(f"Directory absolute path: {os.path.abspath(self.dir)}")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{self.generate_unique_name(seed, config)}.json")
        if self.is_history_exist():
            self.load()
    
    def is_history_exist(self):
        return os.path.exists(self.path)
    
    def generate_unique_name(self, seed, config):
        config_dict = config.to_dict()

        config_dict.pop('eval_tasks', None)
        config_dict.pop('name', None) 

        config_str = json.dumps(config_dict, sort_keys=True)
        
        combined = f"seed_{seed}_config_{config_str}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16] 
    
    def load(self) -> None:
        with open(self.path, "r") as f:
            data = json.load(f)
            self.responses = data['responses']
            self.initial_observation = data['initial_observation']

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({
                "responses": self.responses,
                "initial_observation": self.initial_observation
            }, f, ensure_ascii=False, indent=2)

    def update_initial_observation(self, observation: str):
        self.initial_observation = observation

    def get_initial_observation(self) -> str:
        return self.initial_observation
    
    def update_response(self, response: Union[str, Dict[str, Any]]):
        self.responses.append(response)

    def get_responses(self) -> List[Union[str, Dict[str, Any]]]:
        return self.responses
