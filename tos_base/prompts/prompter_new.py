import numpy as np
from typing import Optional
from .. import Room, Agent, ActionSequence, EvaluationManager
from ..utils.room_utils import get_room_description
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
)
from .prompts import *
from ..utils.utils import THINK_LABEL, ANSWER_LABEL

class PromptManager:
    @staticmethod
    def system_prompt() -> str:
        return f"Strictly follow the required output format with labels {THINK_LABEL} and {ANSWER_LABEL}."

    # Simple env message helpers
    def invalid_action_message(self) -> str:
        return "Invalid action. You should provide only one final action"

    def invalid_format_message(self) -> str:
        return "Invalid output format."

    def steps_left_message(self, remaining_steps: int) -> str:
        return f"You have a maximum of {remaining_steps} exploration steps left."

    def task_finished_message(self) -> str:
        return "Task finished"

    def get_format_footer(self, is_exploration: bool) -> str:
        # Decide answer hint
        if is_exploration:
            answer_hint = "Actions: [ ... ]"
        else:
            answer_hint = "[your answer (only required answer, no extra text, notes, formatting or anything else)]"

        if self.enable_think:
            think = "[Reasoning for next step.]" if is_exploration else "[Your thoughts on the question]"
            return f"## Output Format\n{THINK_LABEL}\n{think}\n{ANSWER_LABEL}\n{answer_hint}"
        else:
            return f"## Output Format\n{ANSWER_LABEL}\n{answer_hint}"

    def __init__(self, config, np_random: np.random.RandomState, image_handler = None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        self.enable_think = bool(self.config.prompt_config.get('enable_think', True))

    def _is_internvl_model(self) -> bool:
        """Return True if current model is InternVL (e.g., internvl3_5)."""
        model_cfg = self.config.get_model_config()
        model_name = str((model_cfg or {}).get('model_name', '')).lower()
        return 'internvl' in model_name

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            exp_history = None
        ) -> tuple:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        obs = {}
        is_vision, is_active = self.config.render_mode == 'vision', self.config.exp_type == 'active'

        room_desc = get_room_description(room, agent)

        # Keep this section aligned to Prompts.md (coords + bins).
        observation_instructions = PairwiseRelationshipDiscrete.prompt()

        exp_instructions = ActionSequence.get_usage_instructions(is_vision)
        if is_active:
            exp_instructions += (
                "- Exploration Strategy:\n"
                "\t- Achieve complete coverage with the fewest steps;\n"
                "\t- Prefer actions that reveal more unknowns; avoid redundancy"
            )
        else:
            exp_history_str = f"## Exploration History\n{exp_history['obs_str']}"
        images_path = []
        if is_vision:
            images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
            images_path = [self.image_handler.get_image_path('instruction'), self.image_handler.get_image_path('label')]
            if not is_active:
                images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])
                images_path.extend(exp_history['multi_modal_data_paths'])
            obs['multi_modal_data'] = {self.config.image_placeholder: images}

        
        template = INSTRUCTION_TEMPLATE_VISION if is_vision else INSTRUCTION_TEMPLATE_TEXT

        fmt_kwargs = {
            'goal_lines': (
                GOAL_EXPLORATION if is_active else ''
            ),
            'env_rules_header': ENV_RULES_HEADER,
            'format_rules': self.get_format_footer(is_active),
            'observation_instructions': observation_instructions,
            'exp_instructions': exp_instructions,
            'room_info': room_desc,
            'context_footer': (
                "Unless otherwise specified, treat the starting position as origin (0, 0), facing North (+y axis).\n"
                f"You have a maximum of {int(self.config.max_exp_steps)} steps."
            ),
            'exp_history': exp_history_str if not is_active else '',
            'vision_example': (VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder) if is_vision else ''),
        }

        obs_str = template.format(**fmt_kwargs)
        obs['obs_str'] = obs_str
        return obs, images_path
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        q = EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
        return f"{q}\n\n{self.get_format_footer(False)}"
