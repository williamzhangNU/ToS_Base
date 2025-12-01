"""False belief task: detect a changed object (rotation or movement)."""

import numpy as np

from .tasks import retry_generate_question
from ..core.relationship import CardinalBinsAllo, StandardDistanceBins, PairwiseRelationshipDiscrete, OrientationRel
from ..utils.utils import hash
from .direction import DirectionPov




# ---- New task: rotate one oriented object, then ask DirectionPov using it as anchor ----
class FalseBeliefDirectionPov(DirectionPov):
    ACTION_TEMPLATE = (
        "Facing north in one room, you note some objects' orientation:\n{observations}\n\n"
    )
    
    QUESTION_TEMPLATE = (
        "Assume the {anchor_name}'s facing defines local north.\n"
        "Where is {obj_name} relative to {anchor_name}?\n\n"
        "Answer format: direction-bin, distance-bin\n"
        "Example: front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [o for o in self.room.objects if o.has_orientation]
        if len(oriented) < 1:
            raise ValueError("Need >=1 oriented objects for this task")

        # 1) Rotate one oriented object (anchor A)
        anchor = self.np_random.choice(oriented)
        deg = int(self.np_random.choice([90, 180, 270]))
        rotations = {0: [[1, 0], [0, 1]], 90: [[0, -1], [1, 0]], 180: [[-1, 0], [0, -1]], 270: [[0, 1], [-1, 0]]}
        anchor.ori = anchor.ori @ rotations[deg]

        # 2) Observe facing north; report all oriented objects in the same room
        tmp_agent = self.agent.copy()
        tmp_agent.ori = np.array((0, 1))
        rid = int(getattr(anchor, 'room_id', getattr(self.agent, 'room_id', 0)))
        objs = [o for o in self.room.objects if o.has_orientation and int(getattr(o, 'room_id', -1)) == rid and o.name != anchor.name][:4] + [anchor]
        objs.sort(key=lambda o: o.name)
        def facing(o):
            op = OrientationRel.get_relative_orientation(tuple(o.ori), tuple(tmp_agent.ori))
            return OrientationRel.to_string(op, 'ego', 'orientation')
        observations = "\n".join(f"{o.name}: {facing(o)}" for o in objs)

        # 3) Ask DirectionPov with this rotated object as anchor A
        target_candidates = [i for i, o in enumerate(self.room.objects) if o is not anchor]
        target = self.room.objects[int(self.np_random.choice(target_candidates))]
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(target.pos),
            tuple(anchor.pos),
            anchor_ori=tuple(anchor.ori),
            bin_system=CardinalBinsAllo(),
            distance_bin_system=StandardDistanceBins(),
        )
        
        # Store the answer directly (open-ended format)
        question = self.QUESTION_TEMPLATE.format(anchor_name=anchor.name, obj_name=target.name)
        self.eval_data.action = self.ACTION_TEMPLATE.format(observations=observations)
        self.eval_data.question = self.eval_data.action + question
        self.eval_data.answer = (rel.direction.bin_label, rel.dist.bin_label)
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        self.eval_data.kwargs = {"rotated_object": anchor.name, "rotation_degrees": deg}
        return self.eval_data.question


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
            try:
                task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
                print(f"Question: {task.generate_question()}")
                print(f"Answer: {task.answer}")
                
                # Test correct answer
                score, info = EvalTaskType.evaluate_prediction(task_name, task.answer, task.answer, task.choices)
                print(f"Correct Answer Evaluation: {score}, details: {info}")
                assert score == 1.0, f"Failed correct answer test for {task_name}"

                # Test robustness for tuple answer
                if isinstance(task.answer, tuple) and len(task.answer) == 2:
                    # Case insensitivity
                    robust_answer = (task.answer[0].upper(), task.answer[1].upper())
                    score, info = EvalTaskType.evaluate_prediction(task_name, robust_answer, task.answer, task.choices)
                    print(f"Robust Answer (Upper) Evaluation: {score}, details: {info}")
                    
                    # Swapped order
                    swapped_answer = (task.answer[1], task.answer[0])
                    score, info = EvalTaskType.evaluate_prediction(task_name, swapped_answer, task.answer, task.choices)
                    print(f"Robust Answer (Swapped) Evaluation: {score}, details: {info}")

                # Test incorrect answer
                incorrect_answer = ("wrong", "answer")
                score, info = EvalTaskType.evaluate_prediction(task_name, incorrect_answer, task.answer, task.choices)
                print(f"Incorrect Answer Evaluation: {score}, details: {info}")
                assert score < 1.0, f"Failed incorrect answer test for {task_name}"

            except ValueError as e:
                print(f"Skipping seed {seed} for {task_name}: {e}")

    task_names = ['false_belief']
    for task_name in task_names:
        test_task(task_name)
