"""Direction and POV evaluation tasks."""

from typing import Iterable, Tuple, List

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
    task.eval_data.answer = (rel.direction.bin_label, rel.dist.bin_label)
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
        "Stand where the {anchor_name} is and face the way it faces.\n"
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




class BackwardPovEvaluationTask(BaseEvaluationTask):
    """Identify which oriented object matches the described egocentric relation."""

    QUESTION_TEMPLATE = (
        "You are standing at an oriented object's position, facing its direction.\n"
        "You observe that {obj_name} is {relation_text}.\n"
        "Which object are you standing at?\n"
        "Answer format: object name\n"
        "Example: lamp\n"
    )

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
        relation_text = f"{rel.direction.bin_label}, {rel.dist.bin_label}"
        question = self.QUESTION_TEMPLATE.format(obj_name=target.name, relation_text=relation_text)
        self.eval_data.question = question
        self.eval_data.answer = anchor.name
        self.eval_data.choices = []
        self.eval_data.id = hash(question)
        return question


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
