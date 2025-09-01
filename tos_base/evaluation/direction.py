"""Direction and POV evaluation tasks."""

from typing import List, Tuple

from .tasks import BaseEvaluationTask
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
    CardinalBinsEgo,
)


class DirectionEvaluationTask(BaseEvaluationTask):
    """Pairwise discrete direction (allocentric) and perspective-taking (egocentric)."""

    QUESTION_TEMPLATE_DIR = (
        "Your starting facing direction is north.\n"
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

    # ---------- small helpers ----------
    def _fmt(self, d: str, s: str) -> str: return f"{d}, {s}"

    def _labels(self, rel):
        dir_labels = rel.direction.bin_system.LABELS
        dist_labels = rel.dist.bin_system.LABELS
        d_idx, s_idx = rel.direction.bin_id, rel.dist.bin_id
        return dir_labels, dist_labels, d_idx, s_idx

    def _wrap(self, k: int, n: int) -> int: return (k + n) % n
    def _clamp(self, k: int, n: int) -> int: return max(0, min(k, n - 1))

    def _compute_discrete_rel(self, pos1, pos2, bin_system, anchor_ori=None):
        return PairwiseRelationshipDiscrete.relationship(
            tuple(pos1), tuple(pos2),
            anchor_ori=tuple(anchor_ori) if anchor_ori is not None else None,
            bin_system=bin_system
        )

    # ---------- wrong-option generators ----------
    def _gen_hard_options(self, rel) -> List[str]:
        """Small, single-axis mistakes (adjacent dir or adjacent distance)."""
        dir_labels, dist_labels, d_idx, s_idx = self._labels(rel)
        out = []
        # same dir, adjacent distance
        for sk in [-2, -2]:
            s = self._fmt(dir_labels[d_idx], dist_labels[self._clamp(s_idx + sk, len(dist_labels))])
            out.append(s)
        # same distance, adjacent dir
        for dk in [2, -2]:
            s = self._fmt(dir_labels[self._wrap(d_idx + dk, len(dir_labels))], dist_labels[s_idx])
            out.append(s)
        return out

    def _gen_challenging_options(self, rel) -> List[str]:
        """Coupled small errors (dir ±1/±2 and dist ±1)."""
        dir_labels, dist_labels, d_idx, s_idx = self._labels(rel)
        out = []
        for dk in [2, -2]:
            for sk in [-2, -2]:
                s = self._fmt(dir_labels[self._wrap(d_idx + dk, len(dir_labels))],
                              dist_labels[self._clamp(s_idx + sk, len(dist_labels))])
                out.append(s)
        return out

    # ---------- shared choice builder ----------
    def generate_choices(self, rel) -> Tuple[List[str], int]:
        dir_labels, dist_labels, d_idx, s_idx = self._labels(rel)
        assert s_idx >= 0, "Distance bin must be positive"

        correct = self._fmt(rel.direction.bin_label, rel.dist.bin_label)
        choices, seen = [correct], {correct}

        # curated candidates
        wrong_options = self._gen_hard_options(rel) + self._gen_challenging_options(rel)
        self.np_random.shuffle(wrong_options)
        for s in wrong_options:
            if len(choices) == 4: break
            if s not in seen:
                choices.append(s); seen.add(s)

        # pad with random valid pairs if needed
        while len(choices) < 4:
            s = self._fmt(self.np_random.choice(dir_labels),
                          self.np_random.choice(dist_labels))
            if s not in seen:
                choices.append(s); seen.add(s)

        self.np_random.shuffle(choices)
        return choices, choices.index(correct)

    # ---------- shared finalize ----------
    def _finalize(self, template: str, obj_name: str, anchor_obj_name: str,
                  choices: List[str], correct_idx: int) -> str:
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.question = template.format(
            obj_name=obj_name, anchor_obj_name=anchor_obj_name, choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    # ---------- allocentric ----------
    def generate_question(self) -> str:
        n = len(self.room.objects)
        i, j = self.np_random.choice(n, size=2, replace=False)
        obj1, obj2 = self.room.objects[i], self.room.objects[j]
        rel = self._compute_discrete_rel(obj1.pos, obj2.pos, CardinalBinsAllo())
        choices, idx = self.generate_choices(rel)
        return self._finalize(self.QUESTION_TEMPLATE_DIR, obj1.name, obj2.name, choices, idx)


class PovEvaluationTask(DirectionEvaluationTask):
    """POV variant of direction task (reuses base helpers)."""

    def generate_question(self) -> str:
        oriented_idxs = [i for i, o in enumerate(self.room.objects) if o.has_orientation]
        assert oriented_idxs, "No oriented objects for POV"
        anchor_idx = int(self.np_random.choice(oriented_idxs))
        obj_idx = int(self.np_random.choice([i for i in range(len(self.room.objects)) if i != anchor_idx]))

        obj = self.room.objects[obj_idx]
        anchor = self.room.objects[anchor_idx]

        rel = self._compute_discrete_rel(obj.pos, anchor.pos, CardinalBinsEgo(), anchor_ori=anchor.ori)
        choices, idx = self.generate_choices(rel)
        return self._finalize(self.QUESTION_TEMPLATE_POV, obj.name, anchor.name, choices, idx)

    def _generate_ego_choices(self, rel) -> Tuple[List[str], int]:
        # Deprecated numeric POV generator. Use discrete bins instead.
        return self.generate_choices(rel)