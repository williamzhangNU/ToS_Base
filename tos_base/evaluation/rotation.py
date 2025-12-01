"""Rotation-related evaluation tasks."""

from typing import List, Tuple
import numpy as np
from typing_extensions import override

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..core.relationship import PairwiseRelationship
from ..utils.utils import hash

class RotEvaluationTask(BaseEvaluationTask):
    """Ask the sequence of objects appearing when rotating in place."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "You will perform a full 360-degree rotation by continuously turning {turn_direction} in place.\n"
        "Assume all walls are removed (you can see through walls), so every object is visible.\n"
        "Focus on this set of objects: {object_pool}.\n"
        "List them in the exact order they appear directly ahead while you rotate.\n"
        "If two objects share a bearing, place the nearer one first.\n\n"
        "Answer format: lamp, chair, table (comma-separated order).\n"
        "Example: mug, sofa, plant\n"
    )

    # ---------- helpers ----------
    def _get_object_info(self, obj: Object, turn_dir: str) -> Tuple[float, float]:
        """Get angle and distance for an object relative to agent position and turn direction."""
        bearing = float(PairwiseRelationship.get_bearing_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)))
        distance = float(PairwiseRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value)
        angle = (bearing % 360.0) if turn_dir == "clockwise" else ((-bearing) % 360.0)
        return angle, distance

    def _sorted_pts(self, turn_dir: str) -> List[Tuple[str, float, float]]:
        pts = []
        for o in self.room.objects:
            if not np.array_equal(o.pos, self.agent.pos):
                ang, dist = self._get_object_info(o, turn_dir)
                pts.append((o.name, ang, dist))
        pts.sort(key=lambda x: (x[1], x[2]))  # by angle, tie -> nearer first
        return pts

    def _greedy_from(self, pts, start_idx: int, eps: float) -> Tuple[List[str], List[float]]:
        n = len(pts)
        names, angs = [], []
        last = None
        for t in range(n):  # one full wrap
            j = (start_idx + t) % n
            name, ang, _ = pts[j]
            if last is None or ((ang - last) % 360.0) > eps:
                names.append(name); angs.append(ang); last = ang
        if (angs[0] - angs[-1]) % 360.0 < eps:
            angs.pop(); names.pop()
        # normalize: start from smallest angle (e.g., [90,180,270,45] -> [45,90,180,270] for CW)
        k = int(np.argmin(angs))
        return names[k:] + names[:k], angs[k:] + angs[:k]

    def _gen_valid_sequence(self, turn_dir: str, eps: float) -> List[str]:
        pts = self._sorted_pts(turn_dir)
        assert len(pts) >= 3, "Need at least 3 objects"
        tries, cur_eps = 0, float(eps)
        while tries < 10:
            start = int(self.np_random.integers(0, len(pts)))
            names, angs = self._greedy_from(pts, start, cur_eps)
            if len(names) >= 3:
                return names[:self.np_random.integers(3, min(len(names), 7) + 1)]
            tries += 1
        # fallback: tighten epsilon and try once more
        cur_eps = min(cur_eps, 1.0)
        print(f"[Rotation Task] Fallback: tighten epsilon to {cur_eps}")
        start = int(self.np_random.integers(0, len(pts)))
        names, angs = self._greedy_from(pts, start, cur_eps)
        assert len(names) >= 3, "Increase object count or decrease angle_eps"
        return names[:self.np_random.integers(3, min(len(names), 7) + 1)]

    # ---------- main ----------
    @retry_generate_question
    def generate_question(self) -> str:
        self.turn_direction = self.np_random.choice(["clockwise", "counterclockwise"])
        self.angle_eps = float(self.config.get("angle_eps", 30.0))

        correct_seq = self._gen_valid_sequence(self.turn_direction, self.angle_eps)
        object_pool = ", ".join(sorted(set(correct_seq)))

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            turn_direction=self.turn_direction,
            object_pool=object_pool,
        )
        self.eval_data.answer = correct_seq
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.turn_direction})"


class RotDualEvaluationTask(RotEvaluationTask):
    """Given the appearing sequence, ask the rotation direction. TODO: different sequences in each option"""

    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "You performed a complete 360° rotation in place.\n"
        "During the rotation, these objects appeared directly in front of you in this order:\n"
        "{object_sequence}\n\n"
        "Based on this sequence, in which direction did you rotate?\n\n"
        "Which direction did you rotate? Reply with a single word like clockwise or counterclockwise.\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        self.turn_direction = self.np_random.choice(["clockwise", "counterclockwise"])
        self.angle_eps = float(self.config.get("angle_eps", 30.0))

        correct_seq = self._gen_valid_sequence(self.turn_direction, self.angle_eps)
        object_sequence = ", ".join(correct_seq)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence
        )
        self.eval_data.answer = self.turn_direction
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question


if __name__ == "__main__":
    from ..utils.room_utils import RoomPlotter, RoomGenerator
    from .task_types import EvalTaskType
    from tqdm import tqdm
    import numpy as np

    def test_task(task_name: str):
        print(f"\nTesting task: {task_name}")
        for seed in tqdm(range(0, 1)):
            np_random = np.random.default_rng(seed)
            room, agent = RoomGenerator.generate_room(
                room_size=(30, 30),
                n_objects=10,
                np_random=np_random,
                room_name='room',
                level=2,
                main=6,
            )
            try:
                task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
                print(f"Question: {task.generate_question()}")
                print(f"Answer: {task.answer}")
                
                # Test correct answer
                score, info = EvalTaskType.evaluate_prediction(task_name, task.answer, task.answer, task.choices)
                print(f"Correct Answer Evaluation: {score}, details: {info}")
                assert score == 1.0, f"Failed correct answer test for {task_name}"

                # Test robustness
                if isinstance(task.answer, str):
                    # Case insensitivity
                    robust_answer = task.answer.upper()
                    score, info = EvalTaskType.evaluate_prediction(task_name, robust_answer, task.answer, task.choices)
                    print(f"Robust Answer (Upper) Evaluation: {score}, details: {info}")
                    assert score == 1.0, f"Failed robust answer (Upper) test for {task_name}"
                    
                    # Extra whitespace
                    robust_answer = task.answer.replace(" ", "  ")
                    score, info = EvalTaskType.evaluate_prediction(task_name, robust_answer, task.answer, task.choices)
                    print(f"Robust Answer (Spaces) Evaluation: {score}, details: {info}")
                    assert score == 1.0, f"Failed robust answer (Spaces) test for {task_name}"

                elif isinstance(task.answer, list):
                    # Test list answer with case insensitivity
                    robust_answer = [a.upper() if isinstance(a, str) else a for a in task.answer]
                    score, info = EvalTaskType.evaluate_prediction(task_name, robust_answer, task.answer, task.choices)
                    print(f"Robust Answer (Upper) Evaluation: {score}, details: {info}")
                    assert score == 1.0, f"Failed robust answer (Upper) test for {task_name}"

                # Test incorrect answer
                incorrect_answer = "wrong answer"
                score, info = EvalTaskType.evaluate_prediction(task_name, incorrect_answer, task.answer, task.choices)
                print(f"Incorrect Answer Evaluation: {score}, details: {info}")
                assert score < 1.0, f"Failed incorrect answer test for {task_name}"

            except ValueError as e:
                print(f"Skipping seed {seed} for {task_name}: {e}")

    task_names = ['rot', 'rot_dual']
    for task_name in task_names:
        test_task(task_name)