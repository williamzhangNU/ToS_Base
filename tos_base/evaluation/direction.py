"""Direction and POV evaluation tasks."""

from typing import List, Tuple

from .tasks import BaseEvaluationTask
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    PairwiseRelationship,
    DirPair,
    Dir,
    CardinalBinsAllo,
    CardinalBinsEgo,
    OrientationRel,
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
        "Each choice is \"<direction-bin>, <distance-bin>\" (egocentric).\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def _compute_discrete_rel(self, pos1, pos2, bin_system, anchor_ori=None):
        return PairwiseRelationshipDiscrete.relationship(tuple(pos1), tuple(pos2), anchor_ori=tuple(anchor_ori) if anchor_ori is not None else None, bin_system=bin_system)

    def _choices_for(self, rel):
        choices, correct_idx = self.generate_choices(rel)
        return choices, correct_idx, self.format_choices(choices, correct_idx)

    def generate_question(self) -> str:
        n = len(self.room.objects)
        pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) for i in range(n) for j in range(i + 1, n)]
        self.np_random.shuffle(pairs)
        i, j = pairs[0]
        obj1, obj2 = self.room.objects[i], self.room.objects[j]
        rel = self._compute_discrete_rel(obj1.pos, obj2.pos, CardinalBinsAllo())
        choices, correct_idx, (choices_text, correct_label) = self._choices_for(rel)

        self.eval_data.question = self.QUESTION_TEMPLATE_DIR.format(
            obj_name=obj1.name,
            anchor_obj_name=obj2.name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, rel) -> Tuple[List[str], int]:
        rnd = self.np_random
        # Allocentric only: discrete bins, hard distractors via adjacent-bin transforms
        dir_labels = rel.direction.bin_system.LABELS
        dist_labels = rel.dist.bin_system.LABELS
        d_idx, d_lab = rel.direction.bin_id, rel.direction.bin_label
        s_idx_raw, s_lab = rel.dist.bin_id, rel.dist.bin_label
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

        all_dist_opts = dist_labels
        while len(choices) < 8:
            s = fmt(rnd.choice(dir_labels), rnd.choice(all_dist_opts))
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
        rel = self._compute_discrete_rel(obj_pos, anchor_obj.pos, CardinalBinsEgo(), anchor_ori=anchor_obj.ori)
        choices, correct_idx, (choices_text, correct_label) = self._choices_for(rel)

        self.eval_data.question = self.QUESTION_TEMPLATE_POV.format(
            anchor_obj_name=anchor_name,
            obj_name=obj_name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def _generate_ego_choices(self, rel) -> Tuple[List[str], int]:
        # Deprecated numeric POV generator. Use discrete bins instead.
        return self.generate_choices(rel)