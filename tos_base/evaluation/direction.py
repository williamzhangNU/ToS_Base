"""Direction and POV evaluation tasks."""

from typing import List, Tuple

from .tasks import BaseEvaluationTask
from ..actions.base import BaseAction
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
    EgoFrontBins,
)


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
            new_d_idx = self._wrap(d_idx + dk, len(dir_labels))
            # POV tasks avoid beyond-fov (first and last indices)
            if self.task_type != "pov" or new_d_idx not in (0, len(dir_labels) - 1):
                s = self._fmt(dir_labels[new_d_idx], dist_labels[s_idx])
                out.append(s)
        return out

    def _gen_challenging_options(self, rel) -> List[str]:
        """Coupled small errors (dir ±2 and dist ±2)."""
        dir_labels, dist_labels, d_idx, s_idx = self._labels(rel)
        out = []
        for dk in [2, -2]:
            for sk in [-2, -2]:
                new_d_idx = self._wrap(d_idx + dk, len(dir_labels))
                # POV tasks avoid beyond-fov (first and last indices)
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
    task_type = "pov"
    QUESTION_TEMPLATE_POV = (
        "Imagine you are at the same position and orientation as the {anchor_obj_name}.\n"
        "From this perspective, what is the spatial relationship of the {obj_name}?\n\n"
        "Each choice is \"<direction-bin>, <distance-bin>\" (egocentric).\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )
    def generate_question(self) -> str:
        oriented_idxs = [i for i, o in enumerate(self.room.objects) if o.has_orientation]
        assert oriented_idxs, "No oriented objects for POV"
        
        # Select an anchor object (the perspective we're taking)
        anchor_idx = int(self.np_random.choice(oriented_idxs))
        anchor = self.room.objects[anchor_idx]
        
        # Find target objects that are visible from the anchor's perspective
        visible_target_idxs = [i for i in range(len(self.room.objects)) 
                              if i != anchor_idx and BaseAction._is_visible(anchor, self.room.objects[i])]
        assert visible_target_idxs, "No objects visible from anchor for POV"

        # Select a target object from those visible to the anchor
        target_idx = int(self.np_random.choice(visible_target_idxs))
        target_obj = self.room.objects[target_idx]

        rel = self._compute_discrete_rel(target_obj.pos, anchor.pos, EgoFrontBins(), anchor_ori=anchor.ori) # TODO change to EgoFrontBin
        choices, idx = self.generate_choices(rel)
        return self._finalize(self.QUESTION_TEMPLATE_POV, target_obj.name, anchor.name, choices, idx)

    def _generate_ego_choices(self, rel) -> Tuple[List[str], int]:
        # Deprecated numeric POV generator. Use discrete bins instead.
        return self.generate_choices(rel)


class BackwardPovEvaluationTask(DirectionEvaluationTask):
    """Backward POV task: Given a spatial relationship, determine which object's perspective you're at."""
    task_type = "bwd_pov"

    QUESTION_TEMPLATE_BWD_POV = (
        "You observe that {obj_name} is {spatial_relationship} from your current perspective.\n"
        "Which object are you currently positioned at sharing the same orientation?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        oriented_idxs = [i for i, o in enumerate(self.room.objects) if o.has_orientation]
        assert oriented_idxs, "No oriented objects for backward POV"
        
        # Find target objects that are visible from at least one oriented object
        valid_target_idxs = []
        for target_idx in range(len(self.room.objects)):
            # Check if any oriented object can see this target
            visible_from_oriented = any(
                oriented_idx != target_idx and 
                BaseAction._is_visible(self.room.objects[oriented_idx], self.room.objects[target_idx])
                for oriented_idx in oriented_idxs
            )
            if visible_from_oriented:
                valid_target_idxs.append(target_idx)
        
        assert valid_target_idxs, "No target objects are visible from any oriented objects"
        
        # Select a target object that has at least one oriented object that can see it
        target_idx = int(self.np_random.choice(valid_target_idxs))
        target_obj = self.room.objects[target_idx]
        
        # Find oriented objects that can see the target object (potential anchors)
        visible_anchor_idxs = [i for i in oriented_idxs 
                              if i != target_idx and BaseAction._is_visible(self.room.objects[i], target_obj)]
        
        assert visible_anchor_idxs, "No oriented objects can see the target for backward POV"
        
        # Select the actual anchor (correct answer)
        correct_anchor_idx = int(self.np_random.choice(visible_anchor_idxs))
        correct_anchor = self.room.objects[correct_anchor_idx]
        
        # Compute the spatial relationship from the correct anchor's perspective
        rel = self._compute_discrete_rel(target_obj.pos, correct_anchor.pos, EgoFrontBins(), anchor_ori=correct_anchor.ori)
        spatial_relationship = self._fmt(rel.direction.bin_label, rel.dist.bin_label)
        
        # Generate choices (potential anchor objects)
        choices = [correct_anchor.name]
        seen = {correct_anchor.name}
        
        # Add wrong choices from other oriented objects
        wrong_candidates = [self.room.objects[i].name for i in oriented_idxs 
                           if i != correct_anchor_idx and self.room.objects[i].name not in seen]
        self.np_random.shuffle(wrong_candidates)
        
        for name in wrong_candidates:
            if len(choices) == 4:
                break
            if name not in seen:
                choices.append(name)
                seen.add(name)
        
        # Pad with any remaining objects if needed
        remaining_objects = [obj.name for obj in self.room.objects 
                           if obj.name not in seen and obj.name != target_obj.name]
        self.np_random.shuffle(remaining_objects)
        
        for name in remaining_objects:
            if len(choices) == 4:
                break
            choices.append(name)
            seen.add(name)
        
        # Ensure we have at least 2 choices
        while len(choices) < 2:
            choices.append(f"object_{len(choices)}")
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_anchor.name)
        
        # Format the question
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.question = self.QUESTION_TEMPLATE_BWD_POV.format(
            obj_name=target_obj.name, 
            spatial_relationship=spatial_relationship,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        
        return self.eval_data.question