"""Direction and POV evaluation tasks."""

from typing import Iterable, Tuple, List, Dict
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
        "From a Top-Down map, describe where {obj_name} is relative to {anchor_name}.\n"
        "Answer format: cardinal direction-bin, distance-bin\n"
        "Example: north-west, near\n"
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
        "Answer format: <object>\n"
        "Example: lamp\n"
    )

    def _format_observation(self, obs: dict) -> str:
        raise NotImplementedError

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need an oriented object for backward POV task")
        self.np_random.shuffle(oriented)
        
        anchor = None
        observations = []
        obj_orientations = {}
        
        # Try to find an anchor that has visible objects
        for candidate in oriented:
            obs_list, oris = self._get_ground_truth_observations(candidate)
            if obs_list:
                anchor = candidate
                observations = obs_list
                obj_orientations = oris
                break
                
        if not anchor:
            raise ValueError("No visible objects from available anchors")
            
        # Pick one observation to describe
        target_obs = self.np_random.choice(observations)
        
        observation_text = self._format_observation(target_obs)
        question = self.QUESTION_TEMPLATE.format(observation=observation_text)
        
        self.eval_data.question = question
        
        # Store detailed answer for validation
        object_positions = {obj.name.lower(): tuple(obj.pos) for obj in self.room.all_objects}
        
        all_orientations = {obj.name.lower(): tuple(obj.ori) for obj in self.room.all_objects if obj.has_orientation}
        
        self.eval_data.answer = {
            'answer': anchor.name,
            'final_observation': [target_obs], # Only enforce the one we described
            'object_positions': object_positions,
            'object_orientations': all_orientations
        }
        self.eval_data.choices = []
        # Hash based on question + answer
        self.eval_data.id = hash(json.dumps(anchor.name) + question)
        return question


class BackwardPovTextEvaluationTask(BaseBackwardPovEvaluationTask):
    """Identify which oriented object matches the described egocentric relation (Text)."""

    def _format_observation(self, obs: dict) -> str:
        parts = [f"{obs['direction']}, {obs['distance']}"]
        if obs.get('orientation'):
             parts.append(f"facing {obs['orientation']}")
        
        relation_text = ", ".join(parts)
        return f"{obs['name']} is {relation_text}"


class BackwardPovVisionEvaluationTask(BaseBackwardPovEvaluationTask):
    """Identify which oriented object matches the described egocentric relation (Vision)."""

    def _format_observation(self, obs: dict) -> str:
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
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType
    from tqdm import tqdm

    # Robustness test suggestions:
    # 1. Case insensitivity: Ensure answers like "North, Near" and "north, near" are equivalent.
    # 2. Extra whitespace: "north,  near" should be valid.
    # 3. Component swapping: "near, north" should be valid if order doesn't matter (check specific task logic).
    # 4. Partial matching: Verify strict vs loose matching requirements.

    task_names = ['dir', 'pov', 'bwd_pov_text', 'bwd_pov_vision', 'dir_anchor']
    # task_names = ['dir'] # Uncomment to run only one

    room, agent, np_random = create_and_plot_room(seed=0)
    for task_name in task_names:
        print(f"\nTesting task: {task_name}")
        try:
            task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
            manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

        except ValueError as e:
            print(f"Skipping {task_name}: {e}")
