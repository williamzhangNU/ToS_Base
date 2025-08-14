"""Direction and POV evaluation tasks."""

from typing import List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.relationship import (
    DirPair,
    Dir,
    DirectionRel,
    TotalRelationship,
    DegreeRel,
    DistanceRel,
)


class DirectionEvaluationTask(BaseEvaluationTask):
    """Pairwise direction (allocentric) and perspective-taking (egocentric)."""

    MODE = 'dir'

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
        mode = self.config.get('mode', getattr(self, 'MODE', 'dir'))

        if mode == 'pov':
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
            rel = TotalRelationship.relationship(tuple(obj_pos), tuple(anchor_obj.pos), anchor_ori=tuple(anchor_obj.ori))
            choices, correct_idx = self.generate_choices(rel, perspective='ego')
            choices_text, correct_label = self.format_choices(choices, correct_idx)
            self.eval_data.question = self.QUESTION_TEMPLATE_POV.format(
                anchor_obj_name=anchor_name,
                obj_name=obj_name,
                choices_text=choices_text,
            )
        else:
            n = len(self.room.objects)
            pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) for i in range(n) for j in range(i + 1, n)]
            self.np_random.shuffle(pairs)
            i, j = pairs[0]
            obj1, obj2 = self.room.objects[i], self.room.objects[j]
            rel = TotalRelationship.relationship(tuple(obj1.pos), tuple(obj2.pos))
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

    def generate_choices(self, rel: TotalRelationship, perspective: str) -> Tuple[List[str], int]:
        rnd = self.np_random
        dr = rel.dir
        deg = rel.deg or DegreeRel(0.0)
        dist = rel.dist or DistanceRel(0.0)

        def wrap(a: DegreeRel) -> DegreeRel:
            x = ((a.value + 180) % 360) - 180
            return DegreeRel(float(round(x)))

        def jdeg(a: DegreeRel, small=True) -> DegreeRel:
            step = rnd.choice([10, 15, 20] if small else [30, 45, 60, 90]) * (1 if rnd.random() < 0.5 else -1)
            return wrap(DegreeRel(a.value + step))

        def jdist(d: DistanceRel, small=True) -> DistanceRel:
            f = rnd.choice([0.9, 1.1] if small else [0.75, 1.25, 1.5])
            return DistanceRel(max(0.01, round(d.value * f, 2)))

        def flip_h(d: DirectionRel) -> DirectionRel:
            p = d.pair
            m = {Dir.LEFT: Dir.RIGHT, Dir.RIGHT: Dir.LEFT}
            return DirectionRel(DirPair(m.get(p.horiz, p.horiz), p.vert))

        def flip_v(d: DirectionRel) -> DirectionRel:
            p = d.pair
            m = {Dir.FORWARD: Dir.BACKWARD, Dir.BACKWARD: Dir.FORWARD}
            return DirectionRel(DirPair(p.horiz, m.get(p.vert, p.vert)))

        def rot90(d: DirectionRel) -> DirectionRel:
            p = d.pair
            t = DirectionRel.TRANSFORMS[(-1, 0)]
            h, v = t[p.horiz], t[p.vert]
            if h in (Dir.FORWARD, Dir.BACKWARD) or v in (Dir.RIGHT, Dir.LEFT):
                h, v = v, h
            return DirectionRel(DirPair(h, v))

        def fmt(d: DirectionRel, a: DegreeRel, s: DistanceRel) -> str:
            return f"{d.to_string(perspective)}, {a.to_string()}, {s.to_string()}"

        correct = fmt(dr, wrap(deg), DistanceRel(dist.value))
        choices, seen = [correct], {correct}

        masks = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        rnd.shuffle(masks)

        def make(m):
            dchg, achg, schg = m
            D, A, S = dr, deg, dist
            if dchg:
                if achg:
                    D, A = rot90(D), wrap(DegreeRel(A.value + 90))
                else:
                    D = rnd.choice([flip_h, flip_v])(D)
            if achg and not dchg:
                A = jdeg(A, small=True if rnd.random() < 0.7 else False)
            if schg:
                S = jdist(S, small=True)
            return fmt(D, A, S)

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

    MODE = 'pov'


