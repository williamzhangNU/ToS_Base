"""Rotation-related evaluation tasks."""

from typing import List, Tuple
import numpy as np

from typing_extensions import override

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.relationship import TotalRelationship
from ..actions import MoveAction, RotateAction


class RotEvaluationTask(BaseEvaluationTask):
    """Ask the sequence of objects appearing when rotating in place."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "perform a full 360° rotation by turning {turn_direction} in place.\n"
        "Identify the order in which objects come directly into view.\n\n"
        "Choose the correct sequence:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )
    MOVEMENT_TEMPLATE = ("You moved to the same position as {move_obj_name}.\n")
    TURN_TEMPLATE = ("You turned clockwise {degree} degrees.\n")

    def generate_question(self) -> str:
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])
        if_move = self.config.get('if_move', False)
        if_turn = self.config.get('if_turn', False)

        movement_prompt = ""
        turn_prompt = ""
        if if_move:
            move_obj = self.np_random.choice(self.room.objects)
            movement_prompt = self.MOVEMENT_TEMPLATE.format(move_obj_name=move_obj.name)
            MoveAction(move_obj.name).execute(self.room, self.agent)
        if if_turn:
            degree = self.np_random.choice([90, 180, 270])
            turn_prompt = self.TURN_TEMPLATE.format(degree=degree)
            RotateAction(degree).execute(self.room, self.agent)

        def bearing_deg(obj: Object) -> Tuple[float, float]:
            deg = TotalRelationship.get_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
            angle = (deg % 360.0) if turn_direction == 'clockwise' else ((-deg) % 360.0)
            dist = TotalRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value
            return angle, dist

        objects = [obj for obj in self.room.objects if not np.array_equal(obj.pos, self.agent.pos)]
        objects.sort(key=bearing_deg)
        correct_answer = [obj.name for obj in objects]

        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = movement_prompt + turn_prompt + self.QUESTION_TEMPLATE.format(
            turn_direction=turn_direction,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, correct_answer: List[str]) -> Tuple[List[str], int]:
        correct_answer_str = ", ".join(correct_answer)
        choices = [correct_answer_str]
        for _ in range(3):
            wrong_list = correct_answer.copy()
            assert len(wrong_list) >= 3, "Need at least 3 objects for this task"
            while ", ".join(wrong_list) in choices:
                self.np_random.shuffle(wrong_list)
            choices.append(", ".join(wrong_list))
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer_str)
        return choices, correct_idx

    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.config.get('turn_direction', 'clockwise')})"



class RotDualEvaluationTask(BaseEvaluationTask):
    """Given the appearing sequence, ask the rotation direction."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "you performed a complete 360° rotation in place.\n"
        "During the rotation, these objects appeared directly in front of you in this order:\n"
        "{object_sequence}\n\n"
        "Based on this sequence, in which direction did you rotate?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])

        def bearing_deg(obj: Object) -> Tuple[float, float]:
            deg = TotalRelationship.get_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
            angle = (deg % 360.0) if turn_direction == 'clockwise' else ((-deg) % 360.0)
            dist = TotalRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value
            return angle, dist

        objects = [obj for obj in self.room.objects if not np.array_equal(obj.pos, self.agent.pos)]
        objects.sort(key=bearing_deg)
        object_names = [obj.name for obj in objects]
        object_sequence = ", ".join(object_names)

        choices, correct_idx = self.generate_choices(turn_direction)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, correct_answer: str) -> Tuple[List[str], int]:
        opposite = 'counterclockwise' if correct_answer == 'clockwise' else 'clockwise'
        choices = [correct_answer, opposite]
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer)
        return choices, correct_idx


