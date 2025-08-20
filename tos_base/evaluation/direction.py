"""Direction and POV evaluation tasks."""

from typing import List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.relationship import (
    DirPair,
    Dir,
    PairwiseRelationship,
)


class DirectionEvaluationTask(BaseEvaluationTask):
    """Pairwise direction (allocentric) and perspective-taking (egocentric)."""

    QUESTION_TEMPLATE_DIR = (
        "From a top-down view, what is the spatial relationship of {obj_name} relative to {anchor_obj_name}?\n"
        "Each choice is \"<relation>, <angle>, <distance>\" for A relative to B.\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )
    QUESTION_TEMPLATE_POV = (
        "Imagine you are at the same position and orientation as the {anchor_obj_name}.\n"
        "From this perspective, what is the spatial relationship of the {obj_name}?\n\n"
        "Each choice is \"<relation>, <angle>, <distance>\" relative to YOU.\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        n = len(self.room.objects)
        pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) for i in range(n) for j in range(i + 1, n)]
        self.np_random.shuffle(pairs)
        i, j = pairs[0]
        obj1, obj2 = self.room.objects[i], self.room.objects[j]
        rel = PairwiseRelationship.relationship(tuple(obj1.pos), tuple(obj2.pos))
        choices, correct_idx = self.generate_choices(rel, perspective='allo')
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE_DIR.format(
            obj_name=obj1.name,
            anchor_obj_name=obj2.name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, rel: PairwiseRelationship, perspective: str) -> Tuple[List[str], int]:
        rnd = self.np_random
        bearing = rel.bearing
        dist_v = rel.distance_value

        def wrap_deg(x: float) -> float:
            return float(((x + 180) % 360) - 180)

        def jdeg(v: float, small=True) -> float:
            step = rnd.choice([10, 15, 20] if small else [30, 45, 60, 90]) * (1 if rnd.random() < 0.5 else -1)
            return wrap_deg(v + step)

        def jdist(d: float, small=True) -> float:
            f = rnd.choice([0.9, 1.1] if small else [0.75, 1.25, 1.5])
            return max(0.01, round(d * f, 2))

        def flip_h(p: DirPair) -> DirPair:
            m = {Dir.LEFT: Dir.RIGHT, Dir.RIGHT: Dir.LEFT}
            return DirPair(m.get(p.horiz, p.horiz), p.vert)

        def flip_v(p: DirPair) -> DirPair:
            m = {Dir.FORWARD: Dir.BACKWARD, Dir.BACKWARD: Dir.FORWARD}
            return DirPair(p.horiz, m.get(p.vert, p.vert))

        def rot90(p: DirPair) -> DirPair:
            return PairwiseRelationship.rotate_pair_90(p)

        def fmt(p: DirPair, deg: float, s: float) -> str:
            dir_s = PairwiseRelationship.pair_to_string(p, perspective)
            deg_s = PairwiseRelationship.format_degree(deg)
            return f"{dir_s}, {deg_s}, {PairwiseRelationship.distance_to_string(s)}"

        correct = fmt(rel.dir_pair, wrap_deg(rel.degree), dist_v)
        choices, seen = [correct], {correct}

        masks = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        rnd.shuffle(masks)

        def make(m):
            dchg, achg, schg = m
            P, A, S = rel.dir_pair, rel.degree, dist_v
            if dchg:
                if achg:
                    P, A = rot90(P), wrap_deg(A + 90)
                else:
                    P = rnd.choice([flip_h, flip_v])(P)
            if achg and not dchg:
                A = jdeg(A, small=True if rnd.random() < 0.7 else False)
            if schg:
                S = jdist(S, small=True)
            return fmt(P, A, S)

        for m in masks:
            for _ in range(3):
                s = make(m)
                if s not in seen:
                    choices.append(s)
                    seen.add(s)
                    break
            if len(choices) == 8:
                break

        while len(choices) < 8:
            s = make(rnd.choice(masks))
            if s not in seen:
                choices.append(s)
                seen.add(s)

        rnd.shuffle(choices)
        return choices, choices.index(correct)


class PovEvaluationTask(DirectionEvaluationTask):
    """POV variant of direction task."""

    def generate_question(self) -> str:

        obj_idx = self.np_random.integers(0, len(self.room.objects))
        oriented_indices = [i for i, obj in enumerate(self.room.objects) if obj.has_orientation]
        assert len(oriented_indices) > 0, "No oriented objects for POV"
        anchor_idx = self.np_random.choice(oriented_indices)
        while obj_idx == anchor_idx:
            obj_idx = self.np_random.integers(0, len(self.room.objects))
        obj_name = self.room.objects[obj_idx].name
        anchor_name = self.room.objects[anchor_idx].name
        obj_pos = self.room.get_object_by_name(obj_name).pos
        anchor_obj = self.room.get_object_by_name(anchor_name)
        rel = PairwiseRelationship.relationship(tuple(obj_pos), tuple(anchor_obj.pos), anchor_ori=tuple(anchor_obj.ori))
        choices, correct_idx = self.generate_choices(rel, perspective='ego')
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE_POV.format(
            anchor_obj_name=anchor_name,
            obj_name=obj_name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question