"""Localization task: deduce target direction and orientation."""

from typing import List, Tuple
import numpy as np

from .tasks import SpatialManipulationTaskBase
from ..core.object import Object
from ..core.relationship import DirPair, Dir, DirectionRel, TotalRelationship


class LocalizationEvaluationTask(SpatialManipulationTaskBase):
    """Localize a target object from an informative diagonal position."""

    QUESTION_TEMPLATE = (
        "You observe the room from another view with new location and orientation.\n"
        "{observations}\n"
        "Based on your observations, what is the direction and orientation of the {target_name} from your current perspective?\n\n"
        "Answer format: (<horiz>, <vert>), <orientation>\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        target_obj = self.np_random.choice(self.room.objects)
        target_name = target_obj.name

        candidates = [obj for obj in self.room.objects if obj.name != target_name and obj.has_orientation]
        assert len(candidates) >= 2, "Need at least 2 oriented objects"
        ref_obj1, ref_obj2 = self.np_random.choice(candidates, size=2, replace=False)

        diagonal_pos = self._get_diagonal_position(target_obj, ref_obj1, ref_obj2)
        self._position_agent_at(diagonal_pos)
        observations = self._take_full_observations(neglect_objects=[target_name])

        dir_rel = TotalRelationship.get_direction(tuple(self.agent.pos), tuple(target_obj.pos), anchor_ori=tuple(self.agent.ori))
        dir_str = dir_rel.to_string('ego')
        ori_rel = TotalRelationship.get_orientation(tuple(target_obj.ori), tuple(self.agent.ori))
        correct_answer = [dir_str, ori_rel.to_string('ego', kind='orientation')]

        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            observations=observations,
            target_name=target_name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, correct_answer: Tuple[str, str]) -> Tuple[List[str], int]:
        h_dirs = [Dir.LEFT, Dir.RIGHT, Dir.SAME]
        v_dirs = [Dir.FORWARD, Dir.BACKWARD, Dir.SAME]
        orientations = ['forward', 'backward', 'right', 'left']
        choices = [f'{correct_answer[0]}, {correct_answer[1]}']
        while len(choices) < 4:
            wrong_dir = DirPair(self.np_random.choice(h_dirs), self.np_random.choice(v_dirs))
            choice = f"{DirectionRel.pair_to_string(wrong_dir, perspective='ego')}, {orientations[self.np_random.choice(range(len(orientations)))]}"
            if choice not in choices:
                choices.append(choice)
        self.np_random.shuffle(choices)
        correct_idx = choices.index(f'{correct_answer[0]}, {correct_answer[1]}')
        return choices, correct_idx


