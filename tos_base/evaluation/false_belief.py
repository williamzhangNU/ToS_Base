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
