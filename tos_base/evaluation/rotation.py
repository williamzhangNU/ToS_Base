"""Rotation-related evaluation tasks."""

from typing import List, Tuple
import numpy as np

from typing_extensions import override

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.relationship import TotalRelationship
from ..actions import MoveAction, RotateAction

"""TODO: 
1. need to debug for 1) multiple front objects 2) very close objects 3) Handle clusters <= 2
2. Change difficulty

"""
class RotEvaluationTask(BaseEvaluationTask):
    """Ask the sequence of objects appearing when rotating in place."""

    QUESTION_TEMPLATE = (
        "You will perform a full 360-degree rotation by continuously turning {turn_direction} in place.\n"
        "Your task is to answer the sequence of objects that will appear in front of you during the rotation.\n"
        "We consider only a randomly chosen subset of objects.\n"
        "If two objects have the exact same bearing, list the nearer first.\n\n"
        "Choose the correct sequence:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )
    MOVEMENT_TEMPLATE = "You moved to the same position as {move_obj_name}.\n"
    TURN_TEMPLATE = "You turned clockwise {degree} degrees.\n"

    def generate_question(self) -> str:
        rnd = self.np_random
        turn_direction = rnd.choice(['clockwise', 'counterclockwise'])
        if_move = self.config.get('if_move', False)
        if_turn = self.config.get('if_turn', False)
        angle_eps = float(self.config.get('angle_eps', 5.0)) # change difficulty
        num_choices = int(self.config.get('num_choices', 4))

        movement_prompt = ""
        turn_prompt = ""
        default_prompt = "You return to your starting position facing north"
        if if_move:
            move_obj = rnd.choice(self.room.objects)
            movement_prompt = self.MOVEMENT_TEMPLATE.format(move_obj_name=move_obj.name)
            MoveAction(move_obj.name).execute(self.room, self.agent, move_anyway=True)
        if if_turn:
            degree = int(rnd.choice([90, 180, 270]))
            turn_prompt = self.TURN_TEMPLATE.format(degree=degree)
            RotateAction(degree).execute(self.room, self.agent)
        state_prompt = default_prompt if not if_move and not if_turn else movement_prompt + turn_prompt

        def measure(o):
            deg = TotalRelationship.get_degree(tuple(o.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
            ang = (deg % 360.0) if turn_direction == 'clockwise' else ((-deg) % 360.0)
            dist = TotalRelationship.get_distance(tuple(o.pos), tuple(self.agent.pos)).value
            return ang, dist

        pool = [o for o in self.room.objects if not np.array_equal(o.pos, self.agent.pos)]
        assert len(pool) >= 3, "Need at least 3 objects for this task"
        pts = [(o.name,)+measure(o) for o in pool]
        pts.sort(key=lambda x: (x[1], x[2]))  # by angle; exact-angle ties near-first

        # build ordered clusters: [{'angle': a0, 'items': [(name, ang, dist), ...]}, ...]
        clusters, cur, a0 = [], [], None
        for it in pts:
            _, ang, _ = it
            if a0 is None or abs(ang - a0) <= angle_eps:
                cur.append(it); a0 = ang if a0 is None else a0
            else:
                clusters.append({'angle': a0, 'items': cur}); cur, a0 = [it], ang
        clusters.append({'angle': a0, 'items': cur})

        choices, correct_idx = self.generate_choices(pts, clusters, num_choices)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.question = state_prompt + self.QUESTION_TEMPLATE.format(
            turn_direction=turn_direction, angle_eps=int(angle_eps), choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, pts, clusters, num_choices):
        rnd = self.np_random

        # Map name -> cluster index
        object2rank = {name: i for i, (name, _, _) in enumerate(pts)}

        def is_wrong_seq(seq):
            """True if cluster indices not strictly increasing."""
            return any(object2rank[seq[i]] >= object2rank[seq[i+1]] for i in range(len(seq)-1))

        def pick_cluster(kmin=2):
            m = int(rnd.integers(kmin, min(len(clusters), 7) + 1))
            idx = sorted(rnd.choice(np.arange(len(clusters)), size=m, replace=False).tolist())
            return [clusters[i] for i in idx]
        
        def expand_cluster(cluster_seq):
            seq = []
            for C in cluster_seq:
                items = C['items']  # already ordered
                r = int(rnd.integers(1, min(len(items), 3) + 1))
                ids = sorted(rnd.choice(np.arange(len(items)), size=r, replace=False).tolist())
                seq += [items[i][0] for i in ids]
            return seq

        # force-wrong ops: swap cluster order
        def swap_clusters(seq):
            assert len(seq) >= 2, "Need at least 2 clusters for this task"
            a, b = rnd.choice(len(seq), size=2, replace=False)
            a, b = max(a, b), min(a, b)
            t = seq[:]; t[a], t[b] = t[b], t[a]
            return t

        # correct
        correct_expanded_seq = expand_cluster(pick_cluster())
        correct = ", ".join(correct_expanded_seq)
        choices, seen = [correct], {correct}

        while len(choices) < num_choices:
            wrong_seq = swap_clusters(pick_cluster())
            expanded_seq = expand_cluster(wrong_seq)
            s = ", ".join(expanded_seq)
            if s not in seen and is_wrong_seq(expanded_seq):
                choices.append(s); seen.add(s)

        rnd.shuffle(choices)
        return choices, choices.index(correct)

    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.config.get('turn_direction', 'clockwise')})"


    
class RotDualEvaluationTask(BaseEvaluationTask):
    """Given the appearing sequence, ask the rotation direction."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "you performed a complete 360° rotation in place.\n"
        "During the rotation, these objects appeared directly in front of you in this order:\n"
        "{object_sequence}\n\n"
        "Based on this sequence, in which direction did you rotate?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])

        def bearing_deg(obj: Object) -> Tuple[float, float]:
            deg = TotalRelationship.get_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
            angle = (deg % 360.0) if turn_direction == 'clockwise' else ((-deg) % 360.0)
            dist = TotalRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value
            return angle, dist

        objects = [obj for obj in self.room.objects if not np.array_equal(obj.pos, self.agent.pos)]
        objects.sort(key=bearing_deg)
        object_names = [obj.name for obj in objects]
        object_sequence = ", ".join(object_names)

        choices, correct_idx = self.generate_choices(turn_direction)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, correct_answer: str) -> Tuple[List[str], int]:
        opposite = 'counterclockwise' if correct_answer == 'clockwise' else 'clockwise'
        choices = [correct_answer, opposite]
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer)
        return choices, correct_idx


class RotMultiStepDirectionTask(BaseEvaluationTask):
    """
    Given step angles (hidden directions) and the front-appearance order of a random subset of objects,
    choose the per-step directions (CW/CCW). Multiple steps allowed; each step angle from a small set;
    total net rotation constrained to ~+360° CW; small CCW "backtracks" allowed but bounded to preserve identifiability.
    """

    QUESTION_TEMPLATE = (
        "You are at the start pose (initial facing unspecified).\n"
        "A subset of objects will be considered.\n"
        "You performed a sequence of in-place rotations with the following step angles (in degrees):\n"
        "{angles}\n\n"
        "During the motion, objects appeared directly in front of you in this order:\n"
        "{sequence}\n\n"
        "What is the step-by-step rotation DIRECTIONS?\n"
        "Use 'CW' for clockwise and 'CCW' for counterclockwise. Example format: CW, CCW, CW\n\n"
        "Choices:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    # Simple, small, safe defaults
    ANGLE_CANDIDATES = [60, 120, 180]    # you can extend if needed
    MIN_OBJECTS = 4                      # subset size lower bound
    MAX_OBJECTS = 7                      # subset size upper bound
    MIN_GAP_DEG = 10.0                   # minimal separation between object bearings
    BOUNDARY_EPS = 2.0                   # safe distance from any step boundary
    MAX_CCW_PULL = 8.0                   # max total CCW backtrack (deg) to avoid order flips
    NUM_CHOICES = 4

    def generate_question(self) -> str:
        rnd = self.np_random

        # 1) pick subset with safe angular separations
        objs = self._sample_object_subset_with_gaps()

        # 2) choose step angles (unknown directions), ensure net ~ +360 with small CCW allowance
        angles = self._sample_angles_sum_360()

        # 3) choose a concrete direction pattern, bounded CCW backtrack; retry until valid
        for _ in range(100):
            dirs = self._sample_directions(len(angles))
            if self._is_valid_direction_plan(angles, dirs, objs):
                break
        else:
            raise RuntimeError("Failed to build a valid multi-step plan.")

        # 4) simulate visible sequence with the true plan
        sequence = self._simulate_appearance_sequence(angles, dirs, objs)

        # 5) build choices: correct + wrong (flip one/two steps), keep only those producing different sequences
        candidates = self._generate_direction_distractors(angles, dirs, objs, want=self.NUM_CHOICES - 1)
        all_choices = [self._dirs_to_str(dirs)] + [self._dirs_to_str(x) for x in candidates]
        rnd.shuffle(all_choices)
        correct_idx = all_choices.index(self._dirs_to_str(dirs))
        choices_text, correct_label = self.format_choices(all_choices, correct_idx)

        # 6) record
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            angles=", ".join(str(a) for a in angles),
            sequence=", ".join(sequence),
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = all_choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    # ---------- helpers ----------

    def _bearing_of(self, obj):
        deg = TotalRelationship.get_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
        return deg % 360.0

    def _dist_of(self, obj):
        return TotalRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value

    def _sample_object_subset_with_gaps(self):
        rnd = self.np_random
        pool = [o for o in self.room.objects if not np.array_equal(o.pos, self.agent.pos)]
        assert len(pool) >= self.MIN_OBJECTS, "Need more objects in the room."
        for _ in range(200):
            k = int(rnd.integers(self.MIN_OBJECTS, min(self.MAX_OBJECTS, len(pool)) + 1))
            subset = rnd.choice(pool, size=k, replace=False).tolist()
            bearings = sorted(self._bearing_of(o) for o in subset)
            if self._has_min_gap(bearings, self.MIN_GAP_DEG):
                return subset
        raise RuntimeError("Failed to sample subset with safe angular gaps.")

    def _has_min_gap(self, bearings, min_gap):
        b = bearings[:]
        for i in range(len(b)):
            a1, a2 = b[i], b[(i+1) % len(b)]
            gap = (a2 - a1) % 360.0
            if gap < min_gap:
                return False
        return True

    def _sample_angles_sum_360(self):
        rnd = self.np_random
        # simple: sample 3~6 steps, each from ANGLE_CANDIDATES, then scale last step to fit near 360 if needed
        for _ in range(200):
            m = int(rnd.integers(3, 7))
            steps = [int(rnd.choice(self.ANGLE_CANDIDATES)) for _ in range(m-1)]
            s = sum(steps)
            last = 360 - s
            if last in self.ANGLE_CANDIDATES:
                return steps + [last]
        # fallback: just three 120's
        return [120, 120, 120]

    def _sample_directions(self, n):
        # random CW/CCW per step; we will validate later
        rnd = self.np_random
        return [rnd.choice(['CW', 'CCW']) for _ in range(n)]

    def _is_valid_direction_plan(self, angles, dirs, objs):
        """
        Enforce:
          1) net rotation in [360-BOUNDARY_EPS, 360+BOUNDARY_EPS]
          2) total CCW backtrack <= MAX_CCW_PULL
          3) no object bearing within BOUNDARY_EPS of any cumulative boundary (to avoid 60 vs 60.1° issues)
        """
        net = 0.0
        ccw_pull = 0.0
        cum = [0.0]
        for a, d in zip(angles, dirs):
            delta = a if d == 'CW' else -a
            net += delta
            if delta < 0: ccw_pull += -delta
            cum.append(net)

        if not (360.0 - self.BOUNDARY_EPS <= net <= 360.0 + self.BOUNDARY_EPS):
            return False
        if ccw_pull > self.MAX_CCW_PULL:
            return False

        # boundary safety: each boundary angle (mod 360) must be far from any object bearing
        boundaries = [x % 360.0 for x in cum]
        obj_bear = [self._bearing_of(o) for o in objs]
        for bd in boundaries:
            for b in obj_bear:
                if min((b - bd) % 360.0, (bd - b) % 360.0) < self.BOUNDARY_EPS:
                    return False
        return True

    def _simulate_appearance_sequence(self, angles, dirs, objs):
        """
        Simulate continuous heading along the piecewise path.
        An object 'appears' each time the unwrapped heading crosses its bearing (mod 360) in the positive direction.
        Because net ~ +360 and CCW pull is small, order is unique.
        If CCW segments exist, they will not be large enough to create alternative crossing orders.
        """
        # precompute bearings/dist; break ties by smaller distance if needed
        items = [(o.name, self._bearing_of(o), self._dist_of(o)) for o in objs]
        items.sort(key=lambda x: (x[1], x[2]))

        seq = []
        cur = 0.0
        for a, d in zip(angles, dirs):
            step = a if d == 'CW' else -a
            nxt = cur + step
            if step >= 0:
                # crossing all bearings in (cur, nxt]
                for name, be, _ in items:
                    if self._crosses(cur, nxt, be):
                        seq.append(name)
            else:
                # CCW backtrack: do nothing (no positive crossing); we only count CW crossings
                pass
            cur = nxt
        # map into one-pass sequence (duplicates unlikely under our constraints)
        return seq

    def _crosses(self, start, end, bearing):
        """
        Positive crossing test on unwrapped line:
        Count a crossing if bearing mod 360 lies in (start_mod360, end_mod360] when moving CW.
        Implement by stepping through every 360 wrap in [start, end].
        """
        s, e = start, end
        # advance window: check every k with s<k*360+bearing<=e
        # Equivalent: exists integer k so that x = k*360 + bearing ∈ (s, e]
        # Compute k range:
        from math import floor, ceil
        k_min = ceil((s - bearing) / 360.0)
        k_max = floor((e - bearing) / 360.0)
        return k_max >= k_min

    def _dirs_to_str(self, dirs):
        return ", ".join(dirs)

    def _generate_direction_distractors(self, angles, true_dirs, objs, want=3):
        """
        Build wrong answers by flipping 1~2 step directions, keep only those that
        produce a different appearance sequence than the question's sequence.
        """
        target_seq = tuple(self._simulate_appearance_sequence(angles, true_dirs, objs))
        res, seen = [], set()
        n = len(true_dirs)

        def add_if_good(dirs):
            key = tuple(dirs)
            if key in seen: return
            if not self._is_valid_direction_plan(angles, dirs, objs): return
            seq = tuple(self._simulate_appearance_sequence(angles, dirs, objs))
            if seq != target_seq:
                res.append(dirs[:]); seen.add(key)

        # flip single
        for i in range(n):
            d = true_dirs[:]
            d[i] = 'CW' if d[i] == 'CCW' else 'CCW'
            add_if_good(d)
            if len(res) >= want: return res

        # flip pairs
        for i in range(n):
            for j in range(i+1, n):
                d = true_dirs[:]
                d[i] = 'CW' if d[i] == 'CCW' else 'CCW'
                d[j] = 'CW' if d[j] == 'CCW' else 'CCW'
                add_if_good(d)
                if len(res) >= want: return res
        return res











if __name__ == "__main__":
    from ragen.env.spatial.Base.tos_base.utils.room_utils import RoomGenerator, RoomPlotter
    
    room, agent = RoomGenerator.generate_room(
        room_size=[20, 20],
        main=5,
        n_objects=5,
        level=3,
        np_random=np.random.default_rng(42),
    )
    RoomPlotter.plot(room, agent, mode='img', save_path='room.png')

    task = RotEvaluationTask(np_random=np.random.default_rng(42), room=room, agent=agent, config={})
    print(task.generate_question())
    print(task.answer)