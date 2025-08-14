"""False belief task: detect a changed object (rotation or movement)."""

from typing import List, Tuple, Any
import numpy as np

from .tasks import SpatialManipulationTaskBase


class FalseBeliefEvaluationTask(SpatialManipulationTaskBase):
    """Identify which object changed and by how much (for rotation) or which moved."""

    ROTATION_TEMPLATE = (
        "One object in the room has rotated.\n"
        "You observe the room from another view with new location and orientation.\n"
        "{observations}\n"
        "Which object rotated and by how many degrees clockwise?\n\n"
        "Answer format: <object_name>, <degrees>\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    MOVEMENT_TEMPLATE = (
        "One object in the room has moved.\n"
        "You observe the room from another view with new location and orientation.\n"
        "{observations}\n"
        "Which object moved?\n\n"
        "Answer format: <object_name>\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        action_type = self.config.get('action_type', 'rotation')
        if action_type == 'movement':
            correct_answer, agent_pos = self._apply_movement()
            template = self.MOVEMENT_TEMPLATE
            self._position_agent_at(agent_pos)
        else:
            correct_answer = self._apply_rotation()
            template = self.ROTATION_TEMPLATE
            self._position_agent_random()

        observations = self._take_full_observations()
        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = template.format(observations=observations, choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def _apply_movement(self) -> Tuple[str, np.ndarray]:
        target_obj = self.np_random.choice(self.room.objects)
        other_objects = [obj for obj in self.room.objects if obj.name != target_obj.name and obj.has_orientation]
        ref_obj1, ref_obj2 = self.np_random.choice(other_objects, size=2, replace=False)
        agent_pos = self._get_diagonal_position(target_obj, ref_obj1, ref_obj2, neglect_trivial=True)
        rel_x, rel_y = target_obj.pos - agent_pos
        for _ in range(200):
            p = self.room.get_random_point(self.np_random)
            if (p[0] - agent_pos[0]) * rel_x >= 0 and (p[1] - agent_pos[1]) * rel_y >= 0:
                target_obj.pos = p
                break
        return target_obj.name, agent_pos

    def _apply_rotation(self) -> Tuple[str, str]:
        oriented_objects = [obj for obj in self.room.objects if obj.has_orientation]
        assert len(oriented_objects) >= 2, "Need at least 2 objects with orientation"
        target_obj = self.np_random.choice(oriented_objects)
        rotation_degrees = self.np_random.choice([90, 180, 270])
        rotations = {90: [[0, -1], [1, 0]], 180: [[-1, 0], [0, -1]], 270: [[0, 1], [-1, 0]]}
        target_obj.ori = target_obj.ori @ rotations[rotation_degrees]
        return target_obj.name, str(rotation_degrees)

    def _position_agent_random(self) -> None:
        self._position_agent_at(self.room.get_random_point(self.np_random))

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        choices = [str(correct_answer) if isinstance(correct_answer, str) else f"{correct_answer[0]}, {correct_answer[1]}"]
        objects = [obj.name for obj in self.room.objects]
        target_len = 4 if self.config.get('action_type', 'rotation') == 'rotation' else len(objects)
        while len(choices) < target_len:
            if isinstance(correct_answer, tuple):
                wrong_obj = self.np_random.choice(objects)
                wrong_deg = self.np_random.choice(['0', '90', '180', '270'])
                choice = f"{wrong_obj}, {wrong_deg}"
            else:
                choice = self.np_random.choice(objects)
            if choice not in choices:
                choices.append(choice)
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer) if isinstance(correct_answer, str) else f"{correct_answer[0]}, {correct_answer[1]}")
        return choices, correct_idx


