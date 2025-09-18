from typing import Optional, List, Dict
import os
import shutil
import json
from ..utils.utils import hash
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
    Directory structure: model_name/room_hash_key/vision_or_text/active_or_passive/
    Example: gpt-4o/1d54fa/vision/active/
    """

    def __init__(self, observation_config:Dict, model_config:Dict ,room_dict: Dict, agent_dict: Dict, output_dir:str, override=False):
        # only explore turn logs are saved
        self.exploration_turn_logs: List[Dict] = []
        self.evaluation_turn_logs: Dict[str, Dict[str, Dict]] = {}
        self.exp_type = observation_config['exp_type']
        self.model_path= HistoryManager.get_model_dir(output_dir, model_config)
        self.output_dir = os.path.abspath(os.path.join(
            self.model_path,
            self._generate_room_key(room_dict, agent_dict),
            observation_config['render_mode'],
            observation_config['exp_type']
        ))
        model_config_path = os.path.join(self.model_path, "model_config.json")
        if observation_config['exp_type'] == 'passive':
            self.output_dir = os.path.join(self.output_dir, observation_config["proxy_agent"])
        self.exploration_path = os.path.join(self.output_dir, "exploration_turn_logs.json")
        self.evaluation_path = os.path.join(self.output_dir, "evaluation_turn_logs.json")
        if override:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)

        self._load()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        if not os.path.exists(model_config_path):
            with open(model_config_path, "w") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=2)


    def is_history_exist(self):
        return os.path.exists(self.exploration_path)
    

    def _generate_room_key(self, room_dict, agent_dict):
        room_str = json.dumps({**room_dict, **agent_dict}, sort_keys=True)

        return hash(room_str)
        
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
                img_path = os.path.join(self.output_dir, "images", f"room_turn_{turn_log['turn_number']}.png")
                RoomPlotter.plot(Room.from_dict(turn_log['room_state']), Agent.from_dict(turn_log['agent_state']), mode='img', save_path=img_path)
                turn_log['room_image'] = img_path
            #invalid
            self.exploration_turn_logs.append(turn_log)
        else:
            assert turn_log['evaluation_log']
            assert turn_log['room_state'] and turn_log['agent_state']

            task_type = turn_log['evaluation_log']['task_type']
            question_id = turn_log['evaluation_log']['evaluation_data']['id']

            img_path = os.path.join(self.output_dir, "images", f"room_{task_type}_{question_id}.png")
            RoomPlotter.plot(Room.from_dict(turn_log['room_state']), Agent.from_dict(turn_log['agent_state']), mode='img', save_path=img_path)
            turn_log['room_image'] = img_path

            # Initialize task type if it doesn't exist
            if task_type not in self.evaluation_turn_logs:
                self.evaluation_turn_logs[task_type] = {}

            # Store the question with its ID
            self.evaluation_turn_logs[task_type][question_id] = turn_log

    def get_responses(self) -> List[Dict]:
        return [log.get('assistant_raw_message') for log in self.exploration_turn_logs if log.get('assistant_raw_message') is not None]


    def update_cogmap(self, turn_log: Dict) -> None:
        """Update cognitive map response for a specific turn"""
        if turn_log['is_exploration_phase']:
            turn_idx = turn_log['turn_number'] - 1
            assert 0 <= turn_idx < len(self.exploration_turn_logs)
            self.exploration_turn_logs[turn_idx]['cogmap_log'] = turn_log['cogmap_log']
        else:
            assert turn_log['evaluation_log']
            assert self.exp_type == 'active'
            task_type = turn_log['evaluation_log']['task_type']
            question_id = turn_log['evaluation_log']['evaluation_data']['id']

            assert task_type in self.evaluation_turn_logs
            assert question_id in self.evaluation_turn_logs[task_type]

            self.evaluation_turn_logs[task_type][question_id]['cogmap_log'] = turn_log['cogmap_log']

    def has_cogmap_response(self, turn_idx: int = None) -> bool:
        """Check if cognitive map response exists for a specific turn (0-indexed)"""
        return (0 <= turn_idx < len(self.exploration_turn_logs) and
                self.exploration_turn_logs[turn_idx].get('cogmap_log'))

    def has_question(self, question_id: str) -> bool:
        """Check if a question with the given ID already exists in evaluation logs"""
        for task_type, questions in self.evaluation_turn_logs.items():
            if question_id in questions:
                return True
        return False

    @staticmethod
    def get_model_dir(output_dir: str, model_config: Dict) -> str:
        """Generate a unique directory name for the model configuration"""
        #TODO may be a minor diff leads to a different hash
        for k in [k for k, v in model_config.items() if v is None]:
            model_config.pop(k)
        model_config.pop("api_key", None) 
        model_config.pop("base_url", None)
        model_config.pop("max_retries", None)
        model_config.pop("timeout", None)
        model_config_str = json.dumps(model_config, sort_keys=True)
        model_name = model_config['model_name'] + "_" + hash(model_config_str)
        return os.path.join(output_dir, model_name)
    
    @staticmethod
    def aggregate_from_directories(model_dir: str, save_images: bool = True) -> Dict:
        """
        Aggregate data from new directory structure:
        base_dir/model_name/hash_value/vision_or_text/active_or_passive/

        Returns:
            Aggregated data dictionary with config_groups organized by text/vision + active/passive combinations
        """
        assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"

        samples = {}
        all_config_keys = set()

        # Scan all sample directories (room keys)
        sample_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

        # Sequential numbering only for valid samples (with valid subdirs)
        valid_idx = 0
        for sample_dir in sample_dirs:
            sample_path = os.path.join(model_dir, sample_dir)

            # Collect subdirectories containing log files
            subdirs: List[str] = []
            for root, _, files in os.walk(sample_path):
                if "exploration_turn_logs.json" in files or "evaluation_turn_logs.json" in files:
                    subdirs.append(root)

            # Skip if subdirs is empty (invalid sample)
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
            for sample_data_dict in samples.values(): # config_key -> sample_data
                if config_name in sample_data_dict and sample_data_dict[config_name] is not None:
                    env_data_list.append(sample_data_dict[config_name])

            if env_data_list:
                result["exp_summary"]["group_performance"][config_name] = ExplorationManager.aggregate_group_performance(env_data_list)
                result["eval_summary"]["group_performance"][config_name] = EvaluationManager.aggregate_group_performance(env_data_list)
                # Provide both exploration and evaluation cogmap summaries
                exp_type = "active" if "active" in config_name else "passive"
                result["cogmap_summary"]["group_performance"][config_name] = CognitiveMapManager.aggregate_group_performance(env_data_list, exp_type=exp_type)
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
            sample_data["env_turn_logs"] = exploration_logs if exploration_logs else [] # Only exploration logs

        # Load evaluation turn logs - store each task separately
        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                evaluation_logs = json.load(f)
            # Store each evaluation task separately
            sample_data["evaluation_tasks"] = evaluation_logs if evaluation_logs else {}

        # Process image paths if save_images is enabled
        if save_images:
            # Process exploration turn logs
            for turn_log in sample_data["env_turn_logs"]:
                if turn_log.get("room_image"):
                    turn_log['room_image'] = os.path.relpath(turn_log['room_image'], model_dir)
                if turn_log.get('message_images'):
                    turn_log['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in turn_log['message_images']]

            # Process evaluation tasks
            for task_questions in sample_data["evaluation_tasks"].values():
                for question_data in task_questions.values():
                    if question_data.get("room_image"):
                        question_data['room_image'] = os.path.relpath(question_data['room_image'], model_dir)
                    if question_data.get('message_images'):
                        question_data['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in question_data['message_images']]

        return sample_data if sample_data["env_turn_logs"] or sample_data["evaluation_tasks"] else None

