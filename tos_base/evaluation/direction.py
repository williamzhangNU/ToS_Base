"""Direction and POV evaluation tasks."""

from typing import List, Tuple

from .tasks import BaseEvaluationTask
from ..actions.base import BaseAction
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
    EgoFrontBins,
)
from ..utils.utils import hash

class DirectionEvaluationTask(BaseEvaluationTask):
    """Pairwise discrete direction (allocentric) and perspective-taking (egocentric)."""
    task_type = "dir"  # 'dir' for allocentric, 'pov' for egocentric

    QUESTION_TEMPLATE_DIR = (
        "Your starting facing direction is north.\n"
        "From a top-down view, what is the spatial relationship of {obj_name} relative to {anchor_obj_name}?\n"
        "Each choice is \"<direction-bin>, <distance-bin>\" (allocentric).\n\n"
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

    def _build_anchor_visible_dict(self):
        """Build dict mapping each oriented object to its visible objects."""
        anchor_visible = {}
        for i, anchor in enumerate(self.room.objects):
            if anchor.has_orientation:
                visible_objects = []
                for j, target in enumerate(self.room.objects):
                    if i != j and BaseAction._is_visible(anchor, target):
                        visible_objects.append((j, target))
                if visible_objects:
                    anchor_visible[i] = visible_objects
        return anchor_visible

    # ---------- wrong-option generators ----------
    def _gen_wrong_options(self, rel) -> List[str]:
        """Generate wrong options: single-axis mistakes and coupled errors."""
        dir_labels, dist_labels, d_idx, s_idx = self._labels(rel)
        out = []
        
        # Single-axis mistakes: same dir, adjacent distance
        for sk in [-2, -2]:
            s = self._fmt(dir_labels[d_idx], dist_labels[self._clamp(s_idx + sk, len(dist_labels))])
            out.append(s)
        
        # Single-axis mistakes: same distance, adjacent dir
        for dk in [2, -2]:
            new_d_idx = self._wrap(d_idx + dk, len(dir_labels))
            if self.task_type != "pov" or new_d_idx not in (0, len(dir_labels) - 1):
                s = self._fmt(dir_labels[new_d_idx], dist_labels[s_idx])
                out.append(s)
        
        # Coupled errors: dir ±1 and dist ±1
        for dk in [1, -1]:
            for sk in [1, -1]:
                new_d_idx = self._wrap(d_idx + dk, len(dir_labels))
                if self.task_type != "pov" or new_d_idx not in (0, len(dir_labels) - 1):
                    s = self._fmt(dir_labels[new_d_idx],
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
        wrong_options = self._gen_wrong_options(rel)
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
        return self.eval_data.question

    # ---------- allocentric ----------
    def generate_question_data(self):
        """Generate question setup data (objects, relationship, etc.)."""
        n = len(self.room.objects)
        i, j = self.np_random.choice(n, size=2, replace=False)
        obj1, obj2 = self.room.objects[i], self.room.objects[j]
        rel = self._compute_discrete_rel(obj1.pos, obj2.pos, CardinalBinsAllo())
        return obj1, obj2, rel, self.QUESTION_TEMPLATE_DIR

    def generate_question(self) -> str:
        while True:
            obj1, obj2, rel, template = self.generate_question_data()
            choices, idx = self.generate_choices(rel)
            question = self._finalize(template, obj1.name, obj2.name, choices, idx)
            self.eval_data.id = hash(question)
            if self.history_manager.has_question(self.eval_data.id):
                continue
            else:
                break
        return question


class PovEvaluationTask(DirectionEvaluationTask):
    """POV variant of direction task (reuses base helpers)."""
    task_type = "pov"
    QUESTION_TEMPLATE_POV = (
        "Imagine you are at the same position and orientation as the {anchor_obj_name}.\n"
        "From this perspective, what is the spatial relationship of the {obj_name}?\n\n"
        "Each choice is \"<direction-bin>, <distance-bin>\" (egocentric).\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )
    def generate_question_data(self):
        """Generate POV question setup data."""
        anchor_visible = self._build_anchor_visible_dict()
        
        if anchor_visible:
            # Normal case: select from anchors that have visible objects
            anchor_idx = int(self.np_random.choice(list(anchor_visible.keys())))
            anchor = self.room.objects[anchor_idx]
            target_idx, target_obj = self.np_random.choice(anchor_visible[anchor_idx])
            rel = self._compute_discrete_rel(target_obj.pos, anchor.pos, EgoFrontBins(), anchor_ori=anchor.ori)
            return target_obj, anchor, rel, self.QUESTION_TEMPLATE_POV, False
        else:
            # Fallback: no objects have visible targets, use beyond-fov
            n = len(self.room.objects)
            anchor_idx, target_idx = self.np_random.choice(n, size=2, replace=False)
            anchor, target_obj = self.room.objects[anchor_idx], self.room.objects[target_idx]
            return target_obj, anchor, None, self.QUESTION_TEMPLATE_POV, True

    def generate_choices_pov_fallback(self):
        """Generate choices for POV fallback case (beyond-fov)."""
        dir_labels = EgoFrontBins().LABELS
        beyond_fov_label = 'beyond-fov'
        dist_labels = ['near']  # Use any distance label
        correct = self._fmt(beyond_fov_label, dist_labels[0])
        
        # Generate choices with beyond-fov as correct, others not beyond-fov
        choices = [correct]
        seen = {correct}
        for label in dir_labels:
            if len(choices) == 4: break
            if label != beyond_fov_label:
                choice = self._fmt(label, dist_labels[0])
                if choice not in seen:
                    choices.append(choice)
                    seen.add(choice)
        
        self.np_random.shuffle(choices)
        return choices, choices.index(correct)

    def generate_question(self) -> str:
        while True:
            target_obj, anchor, rel, template, is_fallback = self.generate_question_data()

            if is_fallback:
                choices, idx = self.generate_choices_pov_fallback()
            else:
                choices, idx = self.generate_choices(rel)

            question = self._finalize(template, target_obj.name, anchor.name, choices, idx)
            self.eval_data.id = hash(question)
            if self.history_manager.has_question(self.eval_data.id):
                continue
            else:
                break
        return question


class BackwardPovEvaluationTask(DirectionEvaluationTask):
    """Backward POV task: Given a spatial relationship, determine which object's perspective you're at."""
    task_type = "bwd_pov"

    QUESTION_TEMPLATE_BWD_POV = (
        "You observe that {obj_name} is {spatial_relationship} from your current perspective.\n"
        "Which object are you currently positioned at sharing the same orientation?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question_data(self):
        """Generate backward POV question setup data."""
        anchor_visible = self._build_anchor_visible_dict()
        
        if anchor_visible:
            # Normal case: select a target that's visible from some anchor
            anchor_idx = int(self.np_random.choice(list(anchor_visible.keys())))
            correct_anchor = self.room.objects[anchor_idx]
            target_idx, target_obj = self.np_random.choice(anchor_visible[anchor_idx])
            
            # Compute spatial relationship from correct anchor's perspective
            rel = self._compute_discrete_rel(target_obj.pos, correct_anchor.pos, EgoFrontBins(), anchor_ori=correct_anchor.ori)
            spatial_relationship = self._fmt(rel.direction.bin_label, rel.dist.bin_label)
            
            return target_obj, correct_anchor, spatial_relationship, anchor_idx, False
        else:
            # Fallback: no visible pairs, generate random relationship with "none" as correct answer
            n = len(self.room.objects)
            target_idx, dummy_anchor_idx = self.np_random.choice(n, size=2, replace=False)
            target_obj = self.room.objects[target_idx]
            spatial_relationship = "front, near"  # Random relationship
            
            return target_obj, None, spatial_relationship, -1, True

    def generate_choices(self, target_obj, correct_anchor, anchor_idx, is_fallback):
        """Generate choices for backward POV task."""
        if is_fallback:
            # Correct answer is "none" since no object can see the target
            choices = ["none"]
            seen = {"none"}
            
            # Add some object names as wrong choices
            for obj in self.room.objects:
                if len(choices) == 4: break
                if obj.name not in seen and obj.name != target_obj.name:
                    choices.append(obj.name)
                    seen.add(obj.name)
        else:
            # Generate choices: correct anchor + wrong anchors (that can't see target)
            choices = [correct_anchor.name]
            seen = {correct_anchor.name}
            
            # Add oriented objects that cannot see the target as wrong choices
            for i, obj in enumerate(self.room.objects):
                if len(choices) == 4: break
                if (obj.has_orientation and i != anchor_idx and obj.name not in seen and
                    not BaseAction._is_visible(obj, target_obj)):
                    choices.append(obj.name)
                    seen.add(obj.name)
        
        # Pad with dummy names if needed
        while len(choices) < 2:
            choices.append(f"object_{len(choices)}")
        
        self.np_random.shuffle(choices)
        correct_name = correct_anchor.name if not is_fallback else "none"
        correct_idx = choices.index(correct_name)
        
        return choices, correct_idx

    def generate_question(self) -> str:
        while True:
            target_obj, correct_anchor, spatial_relationship, anchor_idx, is_fallback = self.generate_question_data()
            choices, correct_idx = self.generate_choices(target_obj, correct_anchor, anchor_idx, is_fallback)

            # Format the question
            choices_text, correct_label = self.format_choices(choices, correct_idx)
            self.eval_data.question = self.QUESTION_TEMPLATE_BWD_POV.format(
                obj_name=target_obj.name,
                spatial_relationship=spatial_relationship,
                choices_text=choices_text
            )
            self.eval_data.answer = correct_label
            self.eval_data.choices = choices
            self.eval_data.id = hash(self.eval_data.question)
            if self.history_manager.has_question(self.eval_data.id):
                continue
            else:
                break
        return self.eval_data.question