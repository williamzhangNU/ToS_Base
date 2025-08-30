from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import os
import json
import hashlib
@dataclass
class ChatMessage:
    role: str  # 'assistant' or 'user'
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }


class HistoryManager:
    """Simple conversation history manager.

    - update(text, reward):
        If reward is not None, treats "text" as assistant message and stores reward.
        If reward is None, treats "text" as user message.
    """

    def __init__(self, seed, config , dir = ".cache"):
        self.messages: List[ChatMessage] = []
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
            self.messages = [ChatMessage(**m) for m in data]

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.to_list(), f, indent=2)
            
    def update(self, role ,text: str) -> ChatMessage:
        """Append a new message.

        - reward is not None -> assistant message (store reward)
        - reward is None -> user message
        """
        content = text
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        return msg

    def to_list(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]
