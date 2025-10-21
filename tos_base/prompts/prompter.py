import numpy as np
from typing import Optional
from .. import Room, Agent, ActionSequence, EvaluationManager
from ..utils.room_utils import get_room_description
from ..core.relationship import (
    PairwiseRelationship, 
    PairwiseRelationshipDiscrete, 
    ProximityRelationship, 
    DegreeRel, OrientationRel
)
from .prompts import *
from ..utils.utils import THINK_LABEL, ANSWER_LABEL

class PromptManager:
    @staticmethod
    def system_prompt() -> str:
        return "You are an AI assistant that answers visual questions based on images."

    # Simple env message helpers
    def invalid_action_message(self) -> str:
        return "Invalid action. You should provide only one final action"

    def invalid_format_message(self) -> str:
        return "Invalid output format."

    def steps_left_message(self, remaining_steps: int) -> str:
        return f"You have a maximum of {remaining_steps} exploration steps left."

    def task_finished_message(self) -> str:
        return "Task finished"
    # Dynamic, reusable format blocks
    def _build_format_rules(self, is_exploration: bool) -> str:
        if self.enable_think:
            think = "[Your thoughts on next step actions]" if is_exploration else "[Your thoughts on the question]"
            answer = "Actions: [ ... ]" if is_exploration else "[your answer]"
            fmt = f"{THINK_LABEL}\n{think}\n{ANSWER_LABEL}\n{answer}"
        else:
            answer = "Actions: [ ... ]" if is_exploration else "[your answer]"
            fmt = f"{ANSWER_LABEL}\n{answer}"
        return (
            "!!! IMPORTANT OUTPUT RULES !!!\n"
            "1. You must always output in this format (labels followed by a newline):\n"
            f"   {fmt}\n"
            f"2. Inside {ANSWER_LABEL}, include ONLY the required answer. No extra text, notes, or formatting.\n"
            "   - No bullet points, prose, boxes, calculations, or explanations.\n"
            "3. Any deviation is invalid."
        )

    def get_format_footer(self, is_exploration: bool) -> str:
        # Decide answer hint
        if is_exploration:
            answer_hint = "Actions: [ ... ]"
        else:
            # Special stricter format for InternVL during evaluation
            if self._is_internvl_model():
                answer_hint = "[ONLY the letter (A, B, C, ...)]"
            else:
                answer_hint = "[your answer (only required answer, no extra text, notes, formatting or anything else)]"

        if self.enable_think:
            think = "[Your thoughts on next step actions]" if is_exploration else "[Your thoughts on the question]"
            return f"Strictly follow this format:\n{THINK_LABEL}\n{think}\n{ANSWER_LABEL}\n{answer_hint}"
        else:
            return f"Strictly follow this format:\n{ANSWER_LABEL}\n{answer_hint}"

    def __init__(self, config, np_random: np.random.RandomState, image_handler = None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        self.enable_think = bool(self.config.prompt_config.get('enable_think', True))

    def _is_internvl_model(self) -> bool:
        """Return True if current model is InternVL (e.g., internvl3_5)."""
        model_cfg = self.config.get_model_config()
        model_name = str((model_cfg or {}).get('model_name', '')).lower()
        print(f"Model name: {model_name}")
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

        observation_instructions = (
            PairwiseRelationship.prompt()
            + f"\n{DegreeRel.prompt()}"
            + f"\n{OrientationRel.prompt()}"
            + f"\n{PairwiseRelationshipDiscrete.prompt()}"
        )
        if not is_vision:
            observation_instructions += f"\n{ProximityRelationship.prompt()}"

        if is_active:
            exp_instructions = f"Action Instructions:\n{ActionSequence.get_usage_instructions(is_vision)}"
            exp_instructions += f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
        else:
            exp_history_str = f"## Exploration History\n{exp_history['obs_str']}" 
        if is_vision:
            images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
            images_path = [self.image_handler.get_image_path('instruction'), self.image_handler.get_image_path('label')]
            if not is_active:
                images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])
                images_path.extend(exp_history['multi_modal_data_paths'])
            obs['multi_modal_data'] = {self.config.image_placeholder: images}

        
        template = INSTRUCTION_TEMPLATE_VISION if is_vision else INSTRUCTION_TEMPLATE_TEXT

        fmt_kwargs = {
            'title': 'Spatial Exploration Task' if is_active else 'Spatial Reasoning Task',
            'intro': SHARED_INTRO_TEXT if not is_vision else SHARED_INTRO_VISION,
            'goal_lines': (
                'Goal: **Minimize total COST** while building a complete and accurate map of the environment.'
                if is_active else ''
            ),
            'format_rules': self._build_format_rules(is_active),
            'observation_instructions': observation_instructions,
            'exp_instructions': exp_instructions if is_active else '',
            'room_info': room_desc,
            'multiroom_rules': SHARED_MULTIROOM_RULES,
            'active_rules_extra': ACTIVE_RULES_EXTRA if is_active else '',
            'rules_common': SHARED_RULES_COMMON,
            'exp_history': exp_history_str if not is_active else '',
            'vision_example': (VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder) if is_vision else ''),
        }

        obs_str = template.format(**fmt_kwargs)
        if is_active:
            obs['obs_str'] = obs_str + "\n" + self.get_format_footer(is_active)
        else:
            obs['obs_str'] = obs_str
        return obs, images_path
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
