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
    # Add FORMAT_PROMPT for backward compatibility
    FORMAT_PROMPT = "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text."

    # Add image prompt constants
    TOPDOWN_PROMPT = "\n\nTopdown view: {placeholder}\n{object_info}"
    # OBLIQUE_PROMPT = "\n\nOblique view: {placeholder}\n{object_info}"

    def __init__(self, config, np_random: np.random.RandomState, image_handler = None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random

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
        room_desc = get_room_description(room, agent, with_topdown=self.config.prompt_config['topdown'])
        if self.config.render_mode == 'vision':
            images = [self.image_handler.get_image('instruction')]
        else:
            observation_instructions = (
                PairwiseRelationship.prompt()
                + f"\n{DegreeRel.prompt()}"
                + f"\n{OrientationRel.prompt()}"
                + f"\n{PairwiseRelationshipDiscrete.prompt()}"
                + f"\n{ProximityRelationship.prompt()}"
            )
        if self.config.exp_type == 'active':
            exp_instructions = ActionSequence.get_usage_instructions() + f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."   
            if self.config.render_mode == 'vision':
                if self.config.prompt_config['topdown']:
                    room_desc += self._get_topdown_prompt(self.TOPDOWN_PROMPT, room)
                    images.append(self.image_handler.get_image('topdown'))

                obs_str = ACTIVE_INSTRUCTION_VISION.format(
                    room_info=room_desc,
                    exp_instructions=exp_instructions,
                    instruction_example=self.config.image_placeholder
                )

                obs['multi_modal_data'] = {self.config.image_placeholder: images}
            else:
                obs_str = ACTIVE_INSTRUCTION_TEXT.format(
                    room_info=room_desc,
                    exp_instructions=exp_instructions,
                    observation_instructions=observation_instructions,
                )

        else:
            exp_history_str = f"## Exploration History\n{exp_history['obs_str']}" if not self.config.prompt_config["topdown"] else ""
            if self.config.render_mode == 'vision':
                if self.config.prompt_config['topdown']:
                    images.append(self.image_handler.get_image('topdown'))
                else:
                    images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])

                obs_str = PASSIVE_INSTRUCTION_VISION.format(
                    room_info=room_desc,
                    exp_history=exp_history_str,
                    instruction_example=self.config.image_placeholder
                )
                obs['multi_modal_data'] = {self.config.image_placeholder: images}
            else:
                obs_str = PASSIVE_INSTRUCTION_TEXT.format(
                    room_info=room_desc,
                    exp_history=exp_history,
                    observation_instructions=observation_instructions,
                )

            obs_str += f"\n{self.get_evaluation_prompt(eval_manager)}"

        obs['obs_str'] = obs_str + "\n" + self.FORMAT_PROMPT
        return obs
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
