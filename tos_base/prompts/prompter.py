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

class Prompter:
    # Strict format prompts
    FORMAT_PROMPT_THINK = "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer>. You must strictly follow this format with no extra text."
    FORMAT_PROMPT_ANSWER_ONLY = "Always output: <answer> [your answer] </answer>. You must strictly follow this format with no extra text."

    # Add image prompt constants
    TOPDOWN_PROMPT = "\n\nTopdown view: {placeholder}\n{object_info}"
    # OBLIQUE_PROMPT = "\n\nOblique view: {placeholder}\n{object_info}"

    def __init__(self, config, np_random: np.random.RandomState, image_handler = None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        self.enable_think = bool(self.config.prompt_config.get('enable_think', True))
        self.FORMAT_PROMPT = self.FORMAT_PROMPT_THINK if self.enable_think else self.FORMAT_PROMPT_ANSWER_ONLY

    def _get_topdown_prompt(self, prompt_template: str, room) -> str:
        """Generate topdown view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def _get_oblique_prompt(self, prompt_template: str, room) -> str:
        """Generate oblique view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            eval_manager: Optional[EvaluationManager] = None,
            exp_history = None
        ) -> dict:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        obs = {}
        is_vision, is_active = self.config.render_mode == 'vision', self.config.exp_type == 'active'
        topdown = self.config.prompt_config['topdown']

        room_desc = get_room_description(room, agent, with_topdown=topdown)

        observation_instructions = (
            PairwiseRelationship.prompt()
            + f"\n{DegreeRel.prompt()}"
            + f"\n{OrientationRel.prompt()}"
            + f"\n{PairwiseRelationshipDiscrete.prompt()}"
        )
        if not is_vision:
            observation_instructions += f"\n{ProximityRelationship.prompt()}"

        exp_instructions = ActionSequence.get_usage_instructions()
        if is_active:
            exp_instructions += f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."

        images = None
        if is_vision:
            images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
            if is_active and topdown:
                room_desc += self._get_topdown_prompt(self.TOPDOWN_PROMPT, room)
                images.append(self.image_handler.get_image('topdown'))
            if not is_active:
                if topdown:
                    images.append(self.image_handler.get_image('topdown'))
                else:
                    images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])

        exp_history_str = ""
        if not is_active:
            exp_history_str = f"## Exploration History\n{exp_history['obs_str']}" if not topdown else ""

        template = (
            ACTIVE_INSTRUCTION_VISION if is_active and is_vision else
            ACTIVE_INSTRUCTION_TEXT if is_active else
            PASSIVE_INSTRUCTION_VISION if is_vision else
            PASSIVE_INSTRUCTION_TEXT
        )

        fmt_kwargs = {
            'room_info': room_desc,
            'observation_instructions': observation_instructions,
            'exp_instructions': exp_instructions,
        }
        if not is_active:
            fmt_kwargs['exp_history'] = exp_history_str
        if is_vision:
            fmt_kwargs['image_placeholder'] = self.config.image_placeholder

        obs_str = template.format(**fmt_kwargs)
        if not is_active:
            obs_str += f"\n{self.get_evaluation_prompt(eval_manager)}"
        if is_vision:
            obs['multi_modal_data'] = {self.config.image_placeholder: images}

        obs['obs_str'] = obs_str + "\n" + self.FORMAT_PROMPT
        return obs
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
