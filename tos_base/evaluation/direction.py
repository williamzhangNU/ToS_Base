"""Direction and POV evaluation tasks."""

from typing import Iterable, Tuple, List
import json
from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
    EgoFrontBins,
    StandardDistanceBins,
)
from ..actions.base import BaseAction
from ..utils.utils import hash


# ---- shared helpers ----
def _store_relation(task: BaseEvaluationTask, question: str, rel: PairwiseRelationshipDiscrete) -> str:
    """Persist relation answer in eval_data."""
    task.eval_data.question = question
    task.eval_data.answer = f"{rel.direction.bin_label}, {rel.dist.bin_label}"
    task.eval_data.choices = []
    task.eval_data.id = hash(question)
    return question


def _visible_relations(room, anchor, rng) -> Iterable[Tuple[object, PairwiseRelationshipDiscrete]]:
    """Yield visible objects in random order with their ego relations."""
    candidates: List = [obj for obj in room.objects if obj is not anchor]
    rng.shuffle(candidates)
    for obj in candidates:
        if not BaseAction._is_visible(anchor, obj):
            continue
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(obj.pos),
            tuple(anchor.pos),
            anchor_ori=tuple(anchor.ori),
            bin_system=EgoFrontBins(),
            distance_bin_system=StandardDistanceBins(),
        )
        yield obj, rel


class DirectionEvaluationTask(BaseEvaluationTask):
    """Ask allocentric relation between two objects."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "From a top-down map, consider these two objects: {obj_name} and {anchor_name}.\n"
        "Describe where {obj_name} is relative to {anchor_name}.\n"
        "Answer format: direction-bin, distance-bin\n"
        "Example: front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        total = len(self.room.objects)
        if total < 2:
            raise ValueError("Need at least two objects to form a relation")
        objects = list(self.room.objects)
        self.np_random.shuffle(objects)
        obj, anchor = objects[0], objects[1]
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(obj.pos),
            tuple(anchor.pos),
            bin_system=CardinalBinsAllo(),
        )
        question = self.QUESTION_TEMPLATE.format(obj_name=obj.name, anchor_name=anchor.name)
        return _store_relation(self, question, rel)


class PovEvaluationTask(BaseEvaluationTask):
    """Ask egocentric relation from an oriented anchor's perspective."""

    QUESTION_TEMPLATE = (
        "Now you jump to where the {anchor_name} is and face the way it faces.\n"
        "Describe {obj_name}'s egocentric relation.\n"
        "Answer format: direction-bin, distance-bin\n"
        "Example: front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need at least one oriented object for POV task")
        self.np_random.shuffle(oriented)
        anchor = None
        visibles: List[Tuple[object, PairwiseRelationshipDiscrete]] = []
        for candidate in oriented:
            visibles = list(_visible_relations(self.room, candidate, self.np_random))
            if visibles:
                anchor = candidate
                break
        if not anchor:
            raise ValueError("No visible objects from available anchors")
        target, rel = self.np_random.choice(visibles)
        question = self.QUESTION_TEMPLATE.format(anchor_name=anchor.name, obj_name=target.name)
        return _store_relation(self, question, rel)




class BaseBackwardPovEvaluationTask(BaseEvaluationTask):
    """Identify which oriented object matches the described egocentric relation."""

    QUESTION_TEMPLATE = (
        "Now you jump to an oriented object's position, facing its direction.\n"
        "You observe that {observation}.\n"
        "Which object are you standing at?\n"
        "Answer format: object name\n"
        "Example: lamp\n"
    )

    def _get_observation(self, target, rel) -> str:
        raise NotImplementedError

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need an oriented object for backward POV task")
        self.np_random.shuffle(oriented)
        anchor = None
        visibles: List[Tuple[object, PairwiseRelationshipDiscrete]] = []
        for candidate in oriented:
            visibles = list(_visible_relations(self.room, candidate, self.np_random))
            if visibles:
                anchor = candidate
                break
        if not anchor:
            raise ValueError("No visible objects from available anchors")
        target, rel = self.np_random.choice(visibles)
        
        observation = self._get_observation(target, rel)
        question = self.QUESTION_TEMPLATE.format(observation=observation)
        
        self.eval_data.question = question
        self.eval_data.answer = anchor.name
        self.eval_data.choices = []
        self.eval_data.id = hash(json.dumps(self.eval_data.answer) + question)
        return question


class BackwardPovTextEvaluationTask(BaseBackwardPovEvaluationTask):
    """Identify which oriented object matches the described egocentric relation (Text)."""

    def _get_observation(self, target, rel) -> str:
        relation_text = f"{rel.direction.bin_label}, {rel.dist.bin_label}"
        return f"{target.name} is {relation_text}"


class BackwardPovVisionEvaluationTask(BaseBackwardPovEvaluationTask):
    """Identify which oriented object matches the described egocentric relation (Vision)."""

    def _get_observation(self, target, rel) -> str:
        return "You observe: <image>"

class DirectionPov(BaseEvaluationTask):
    """Allocentric relation treating the anchor's facing as north."""

    QUESTION_TEMPLATE = (
        "Assume the {anchor_name}'s facing defines local north.\n"
        "Where is {obj_name} relative to {anchor_name}?\n"
        "Answer format: direction-bin, distance-bin\n"
        "Example: front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need an oriented object for DirectionPov task")
        self.np_random.shuffle(oriented)
        anchor = oriented[0]
        others = [obj for obj in self.room.objects if obj is not anchor]
        if not others:
            raise ValueError("DirectionPov task requires another object besides the anchor")
        self.np_random.shuffle(others)
        target = others[0]
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(target.pos),
            tuple(anchor.pos),
            anchor_ori=tuple(anchor.ori),
            bin_system=CardinalBinsAllo(),
        )
        question = self.QUESTION_TEMPLATE.format(anchor_name=anchor.name, obj_name=target.name)
        return _store_relation(self, question, rel)



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
            # RoomPlotter.plot(room, agent, mode='img', save_path=f'room_{task_name}.png')
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

                    # Swapped order (for direction tasks)
                    if "," in task.answer:
                        parts = [p.strip() for p in task.answer.split(",")]
                        if len(parts) == 2:
                            swapped_answer = f"{parts[1]}, {parts[0]}"
                            score, info = EvalTaskType.evaluate_prediction(task_name, swapped_answer, task.answer, task.choices)
                            print(f"Robust Answer (Swapped) Evaluation: {score}, details: {info}")
                            assert score == 1.0, f"Failed robust answer (Swapped) test for {task_name}"

                # Test incorrect answer
                incorrect_answer = "wrong answer"
                score, info = EvalTaskType.evaluate_prediction(task_name, incorrect_answer, task.answer, task.choices)
                print(f"Incorrect Answer Evaluation: {score}, details: {info}")
                assert score < 1.0, f"Failed incorrect answer test for {task_name}"

            except ValueError as e:
                print(f"Skipping seed {seed} for {task_name}: {e}")

    task_names = ['dir', 'pov', 'bwd_pov_text', 'bwd_pov_vision', 'dir_anchor']
    for task_name in task_names:
        test_task(task_name)
