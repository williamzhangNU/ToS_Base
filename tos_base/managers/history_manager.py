from typing import Optional, List, Dict
import os
import shutil
import json
import hashlib
from ..utils.room_utils import RoomPlotter
from .. import (
    Agent,
    Room,
    EvaluationManager,
    ExplorationManager,
    CognitiveMapManager,
)
class HistoryManager:
    """Simple conversation history manager.
    Store only env turn logs in a single JSON file
    """

    def __init__(self, observation_config:Dict, room_dict: Dict, agent_dict: Dict, override=False, dir = "results/debug"):
        # dir structure model_name/room_key/vision_or_text/active_or_passive/
        # only explore turn logs are saved
        self.exploration_turn_logs: List[Dict] = []
        self.evaluation_turn_logs: Dict = {}
        self.exp_type = observation_config['exp_type']
        self.dir = os.path.abspath(os.path.join(dir, observation_config['model_name'], self._generate_room_key(room_dict, agent_dict), observation_config['render_mode'], observation_config['exp_type']))
        if observation_config['exp_type'] == 'passive':
            self.dir = os.path.join(self.dir, observation_config["proxy_agent"])
        self.exploration_path = os.path.join(self.dir, "exploration_turn_logs.json")
        self.evaluation_path = os.path.join(self.dir, "evaluation_turn_logs.json")
        if override:
            if os.path.exists(self.dir):
                shutil.rmtree(self.dir)

        self._load()
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
        
    
    def is_history_exist(self):
        return os.path.exists(self.exploration_path)
    
    def _generate_room_key(self, room_dict, agent_dict):
        room_str = json.dumps({**room_dict, **agent_dict}, sort_keys=True)

        return hashlib.sha256(room_str.encode("utf-8")).hexdigest()[:16]
        
    def _load(self):
        """Load env turn logs from JSON file"""
        if os.path.exists(self.exploration_path):
            with open(self.exploration_path, "r") as f:
                self.exploration_turn_logs = json.load(f)
        if os.path.exists(self.evaluation_path):
            with open(self.evaluation_path, "r") as f:
                self.evaluation_turn_logs = json.load(f)
    
    def save(self) -> None:
        """Save env turn logs to JSON file"""
        if self.exploration_turn_logs:
            with open(self.exploration_path, "w") as f:
                json.dump(self.exploration_turn_logs, f, ensure_ascii=False, indent=2)
        with open(self.evaluation_path, "w") as f:
            json.dump(self.evaluation_turn_logs, f, ensure_ascii=False, indent=2)
    
    def update_turn_log(self, turn_log: Dict):
        """
            Add or update a turn log with all necessary data including images
            Must be added in sequence
        """
        if turn_log['is_exploration_phase']:
            assert not self.is_history_exist()
            if turn_log['room_state'] and turn_log['agent_state']:
                img_path = os.path.join(self.dir, "images", f"room_turn_{turn_log['turn_number']}.png")
                RoomPlotter.plot(Room.from_dict(turn_log['room_state']), Agent.from_dict(turn_log['agent_state']), mode='img', save_path=img_path)
                turn_log['room_image'] = img_path
            #invalid
            self.exploration_turn_logs.append(turn_log)
        else:
            assert turn_log['evaluation_log']
            assert turn_log['room_state'] and turn_log['agent_state']
            img_path = os.path.join(self.dir, "images", f"room_{turn_log['evaluation_log']['task_type']}.png")
            RoomPlotter.plot(Room.from_dict(turn_log['room_state']), Agent.from_dict(turn_log['agent_state']), mode='img', save_path=img_path)
            turn_log['room_image'] = img_path
            self.evaluation_turn_logs[turn_log['evaluation_log']['task_type']] = turn_log


    def get_responses(self) -> List[Dict]:
        return [log.get('assistant_raw_message') for log in self.exploration_turn_logs if log.get('assistant_raw_message') is not None]


    def update_cogmap(self, turn_log: Dict) -> None:
        """Update cognitive map response for a specific turn"""
        if turn_log['is_exploration_phase']:
            turn_idx = turn_log['turn_number'] - 1
            assert 0 <= turn_idx < len(self.exploration_turn_logs)
            self.exploration_turn_logs[turn_idx]['cogmap_response'] = turn_log['cogmap_response']
            self.exploration_turn_logs[turn_idx]['cogmap_log'] = turn_log['cogmap_log']
            self.exploration_turn_logs[turn_idx]['cogmap_full_log'] = turn_log['cogmap_full_log']
        else:
            assert turn_log['evaluation_log']
            assert self.exp_type == 'active'
            task_type = turn_log['evaluation_log']['task_type']
            assert task_type in self.evaluation_turn_logs
            self.evaluation_turn_logs[task_type]['cogmap_response'] = turn_log['cogmap_response']
            self.evaluation_turn_logs[task_type]['cogmap_log'] = turn_log['cogmap_log']
            self.evaluation_turn_logs[task_type]['cogmap_full_log'] = turn_log['cogmap_full_log']

    def has_cogmap_response(self, turn_idx: int = None) -> bool:
        """Check if cognitive map response exists for a specific turn (0-indexed)"""
        return (0 <= turn_idx < len(self.exploration_turn_logs) and
                self.exploration_turn_logs[turn_idx].get('cogmap_response') is not None)

    @staticmethod
    def _aggregate_from_directories(model_dir: str, save_images: bool = True) -> Dict:
        """
        Aggregate data from new directory structure:
        base_dir/model_name/hash_value/vision_or_text/active_or_passive/

        Returns:
            Aggregated data dictionary with config_groups organized by text/vision + active/passive combinations
        """
        assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"

        samples = {}
        all_config_keys = set()

        # Scan all sample directories (hash values)
        sample_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

        # 连续编号仅针对有效样本（存在合法 subdirs）
        valid_idx = 0
        for sample_dir in sample_dirs:
            sample_path = os.path.join(model_dir, sample_dir)

            # 收集包含日志文件的子目录
            subdirs: List[str] = []
            for root, _, files in os.walk(sample_path):
                if "exploration_turn_logs.json" in files or "evaluation_turn_logs.json" in files:
                    subdirs.append(root)

            # subdirs 为空则视为无效样本，跳过
            if not subdirs:
                continue

            valid_idx += 1
            sample_key = f"sample_{valid_idx}"
            samples[sample_key] = {}

            for combo_path in subdirs:
                rel = os.path.relpath(combo_path, sample_path)
                if rel in (".", ""):
                    continue
                config_key = rel.replace(os.sep, "_")

                sample_data = HistoryManager._load_sample_data(
                    combo_path=combo_path,
                    sample_key=sample_key,
                    save_images=save_images,
                    model_dir=model_dir,
                )
                if sample_data:
                    samples[sample_key][config_key] = sample_data
                    all_config_keys.add(config_key)

        # Initialize result structure with samples
        result = {
            "samples": samples,
            "exp_summary": {"group_performance": {}},
            "eval_summary": {"group_performance": {}},
            "cogmap_summary": {"group_performance": {}}
        }

        # Aggregate performance for each config combination across all samples
        for config_name in sorted(all_config_keys):
            env_data_list = []
            for sample_data_dict in samples.values():
                if config_name in sample_data_dict and sample_data_dict[config_name] is not None:
                    env_data_list.append(sample_data_dict[config_name])

            if env_data_list:
                result["exp_summary"]["group_performance"][config_name] = ExplorationManager.aggregate_group_performance(env_data_list)
                result["eval_summary"]["group_performance"][config_name] = EvaluationManager.aggregate_group_performance(env_data_list)
                result["cogmap_summary"]["group_performance"][config_name] = CognitiveMapManager.aggregate_group_performance(env_data_list)

        return result

    @staticmethod
    def _load_sample_data(combo_path: str, sample_key: str, save_images: bool, model_dir: str) -> Optional[Dict]:
        """Load data from a single sample's combination directory"""
        exploration_file = os.path.join(combo_path, "exploration_turn_logs.json")
        evaluation_file = os.path.join(combo_path, "evaluation_turn_logs.json")

        sample_data = {
            "sample_id": sample_key,
            "env_turn_logs": [],  # Only exploration turn logs
            "evaluation_tasks": {},  # Separate storage for evaluation tasks
        }

        # Load exploration turn logs
        if os.path.exists(exploration_file):
            with open(exploration_file, 'r') as f:
                exploration_logs = json.load(f)
                if exploration_logs:
                    sample_data["env_turn_logs"] = exploration_logs  # Only exploration logs

        # Load evaluation turn logs - store each task separately
        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                evaluation_logs = json.load(f)
                if evaluation_logs and isinstance(evaluation_logs, dict):
                    # Store each evaluation task separately
                    sample_data["evaluation_tasks"] = evaluation_logs

        # Process image paths if save_images is enabled
        if save_images:
            # Process exploration turn logs
            for turn_log in sample_data["env_turn_logs"]:
                if turn_log.get("room_image"):
                    turn_log['room_image'] = os.path.relpath(turn_log['room_image'], model_dir)
                if turn_log.get('message_images'):
                    turn_log['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in turn_log['message_images']]

            # Process evaluation tasks
            for task_name, eval_log in sample_data["evaluation_tasks"].items():
                if eval_log.get("room_image"):
                    eval_log['room_image'] = os.path.relpath(eval_log['room_image'], model_dir)
                if eval_log.get('message_images'):
                    eval_log['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in eval_log['message_images']]

        return sample_data if sample_data["env_turn_logs"] or sample_data["evaluation_tasks"] else None

