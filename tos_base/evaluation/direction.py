"""Direction and POV evaluation tasks."""

from typing import List, Tuple

from .tasks import BaseEvaluationTask
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    DirectionRelDiscrete,
    DistanceRelBinned,
    PairwiseRelationship,
    DirPair,
    Dir,
)


class DirectionEvaluationTask(BaseEvaluationTask):
    """Pairwise direction (allocentric) and perspective-taking (egocentric)."""

    QUESTION_TEMPLATE_DIR = (
        "From a top-down view, what is the spatial relationship of {obj_name} relative to {anchor_obj_name}?\n"
        "Each choice is \"<direction-bin>, <distance-bin>\" (allocentric).\n\n"
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
        rel = PairwiseRelationshipDiscrete.relationship(tuple(obj1.pos), tuple(obj2.pos))
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

    def generate_choices(self, rel, perspective: str) -> Tuple[List[str], int]:
        rnd = self.np_random
        # Allocentric: discrete bins, hard distractors via adjacent-bin transforms
        if perspective == 'allo':
            dir_labels = DirectionRelDiscrete.ALLO_LABELS
            dist_labels = DistanceRelBinned.DISTANCE_BIN_LABELS
            zero_label = DistanceRelBinned.DIST_ZERO_LABEL
            d_idx, d_lab = DirectionRelDiscrete.bin_bearing(rel.direction.degree, 'allo')
            s_idx_raw, s_lab = DistanceRelBinned.bin_distance(rel.dist.value)
            s_idx = 0 if s_idx_raw < 0 else s_idx_raw

            def fmt(d: str, s: str) -> str: return f"{d}, {s}"
            def wrap8(k: int) -> int: return (k + 8) % 8
            def clamp(k: int) -> int: return max(0, min(k, len(dist_labels) - 1))

            correct = fmt(d_lab, s_lab)
            choices, seen = [correct], {correct}

            dir_shifts = [1, -1, 2, -2, 4]
            dist_shifts = [-1, 1]

            cands = []
            cands += [fmt(dir_labels[wrap8(d_idx + k)], s_lab) for k in dir_shifts]
            if s_idx_raw >= 0:
                cands += [fmt(d_lab, dist_labels[clamp(s_idx + k)]) for k in dist_shifts]
            else:
                cands += [fmt(d_lab, dist_labels[0])]
            for dk in [1, -1, 2, -2]:
                for sk in [-1, 1]:
                    cands.append(fmt(dir_labels[wrap8(d_idx + dk)], dist_labels[clamp(s_idx + sk)]))

            for s in cands:
                if len(choices) == 8: break
                if s not in seen:
                    choices.append(s); seen.add(s)

            all_dist_opts = dist_labels + [zero_label]
            while len(choices) < 8:
                s = fmt(rnd.choice(dir_labels), rnd.choice(all_dist_opts))
                if s not in seen:
                    choices.append(s); seen.add(s)

            rnd.shuffle(choices)
            return choices, choices.index(correct)

        # Ego: keep numeric relation, degree, distance (unchanged POV style)
        def wrap_deg(x: float) -> float: return float(((x + 180) % 360) - 180)
        def jdeg(v: float) -> float:
            return wrap_deg(v + rnd.choice([10, 15, 20, 30, 45]) * (1 if rnd.random() < 0.5 else -1))
        def jdist(d: float) -> float:
            return max(0.01, round(d * rnd.choice([0.9, 1.1, 0.75, 1.25]), 2))
        def flip_h(p: DirPair) -> DirPair:
            m = {Dir.LEFT: Dir.RIGHT, Dir.RIGHT: Dir.LEFT}
            return DirPair(m.get(p.horiz, p.horiz), p.vert)
        def flip_v(p: DirPair) -> DirPair:
            m = {Dir.FORWARD: Dir.BACKWARD, Dir.BACKWARD: Dir.FORWARD}
            return DirPair(p.horiz, m.get(p.vert, p.vert))
        def rot90(p: DirPair) -> DirPair:
            return PairwiseRelationship.rotate_pair_90(p)
        def fmt_num(p: DirPair, deg: float, s: float) -> str:
            return f"{PairwiseRelationship.pair_to_string(p, 'ego')}, {PairwiseRelationship.format_degree(deg)}, {PairwiseRelationship.distance_to_string(s)}"

        P, A, S = rel.dir_pair, wrap_deg(rel.degree), rel.distance_value
        correct = fmt_num(P, A, S)
        choices, seen = [correct], {correct}

        masks = [(1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 0, 1), (0, 1, 1)]
        rnd.shuffle(masks)
        def make(m):
            dchg, achg, schg = m
            p, a, s = P, A, S
            if dchg:
                if achg: p, a = rot90(p), wrap_deg(a + 90)
                else: p = rnd.choice([flip_h, flip_v])(p)
            if achg and not dchg: a = jdeg(a)
            if schg: s = jdist(s)
            return fmt_num(p, a, s)

        for m in masks:
            for _ in range(3):
                s = make(m)
                if s not in seen:
                    choices.append(s); seen.add(s); break
            if len(choices) == 8: break
        while len(choices) < 8:
            s = make(rnd.choice(masks))
            if s not in seen:
                choices.append(s); seen.add(s)
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