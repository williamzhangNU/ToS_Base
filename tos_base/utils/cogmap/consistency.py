from typing import Dict, Tuple, List
import numpy as np
from itertools import combinations

from ...core.room import BaseRoom, Room
from ...core.object import Agent
from ...core.relationship import PairwiseRelationshipDiscrete, CardinalBinsAllo, StandardDistanceBins
from ...managers.spatial_solver import SpatialSolver
from .transforms import br_from_anchor_to_initial
from .metrics import compute_map_metrics
from .types import MapCogMetrics
from ..relation_codes import decode_relation_codes, make_pair_key, encode_relation_codes


def compare_on_common_subset(a: BaseRoom | None, b: BaseRoom | None, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if a is None or b is None:
        return MapCogMetrics.invalid()
    names_a = {o.name for o in a.objects}
    names_b = {o.name for o in b.objects}
    names = names_a & names_b
    if not names:
        return MapCogMetrics.invalid()
    a_sub = BaseRoom(objects=[o for o in a.objects if o.name in names], name=a.name)
    b_sub = BaseRoom(objects=[o for o in b.objects if o.name in names], name=b.name)
    return compute_map_metrics(a_sub, b_sub, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def local_vs_global_consistency(pred_local: BaseRoom | None, pred_global: BaseRoom | None, agent: Agent, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if pred_local is None or pred_global is None:
        return MapCogMetrics.invalid()
    local_in_initial = br_from_anchor_to_initial(pred_local, np.array(agent.pos, dtype=float), np.array(agent.ori, dtype=int), agent)
    return compare_on_common_subset(local_in_initial, pred_global, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def rooms_vs_global_consistency(pred_rooms: Dict[str, BaseRoom], pred_global: BaseRoom | None, room: Room, agent: Agent, entry_gate_by_room: Dict[int, str], allow_scale: bool, pos_norm_L: float | None) -> Tuple[MapCogMetrics, Dict[str, MapCogMetrics]]:
    if pred_global is None:
        return MapCogMetrics.invalid(), {}
    per_room: Dict[str, MapCogMetrics] = {}
    vals: List[MapCogMetrics] = []
    for rid, room_br in sorted(pred_rooms.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]):
        gate_name = entry_gate_by_room.get(int(rid))
        if not gate_name:
            continue
        g = next((gg for gg in room.gates if gg.name == gate_name), None)
        if g is None:
            continue
        gate_pos = g.pos
        gate_ori = g.get_ori_for_room(int(rid))
        room_in_initial = br_from_anchor_to_initial(room_br, gate_pos, gate_ori, agent)
        m = compare_on_common_subset(room_in_initial, pred_global, allow_scale=allow_scale, pos_norm_L=pos_norm_L)
        if m.valid:
            per_room[rid] = m
            vals.append(m)
    avg = MapCogMetrics.average(vals) if vals else MapCogMetrics.invalid()
    return avg, per_room


def map_vs_relations_consistency(pred_relations: Dict, pred_global: BaseRoom | None) -> float:
    """Compare consistency between global map coordinates and pairwise relations.

    Args:
        pred_relations: Dict with pairwise relations like {"A|B": "(dir_code, dist_code)"}
        pred_global: BaseRoom with object positions, or None

    Returns:
        Float between 0.0 and 1.0 representing consistency (1.0 = perfect match)
    """
    if not pred_relations or pred_global is None:
        return 0.0

    # Extract positions from BaseRoom
    positions = {}
    for obj in pred_global.objects:
        if hasattr(obj, 'pos') and hasattr(obj, 'name'):
            pos = obj.pos
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                positions[obj.name] = (int(pos[0]), int(pos[1]))

    if len(positions) < 2:
        return 0.0

    # Generate expected relations from map positions
    expected_relations = {}
    bin_system = CardinalBinsAllo()
    distance_bin_system = StandardDistanceBins()

    for name_a, name_b in combinations(positions.keys(), 2):
        pos_a, pos_b = positions[name_a], positions[name_b]
        # Create discrete relationship from positions
        rel = PairwiseRelationshipDiscrete.relationship(
            pos_a, pos_b, anchor_ori=(0, 1),
            bin_system=bin_system, distance_bin_system=distance_bin_system
        )
        pair_key = make_pair_key(name_a, name_b)
        expected_relations[pair_key] = encode_relation_codes(
            rel.direction.bin_label, rel.dist.bin_label
        )

    # Compare with predicted relations
    if not expected_relations:
        return 0.0

    matches = 0
    total = len(expected_relations)

    for pair_key, expected_rel in expected_relations.items():
        predicted_rel = pred_relations.get(pair_key, "")
        if predicted_rel == expected_rel:
            matches += 1
        else:
            # Check partial matches (direction or distance)
            exp_dir, exp_dist = decode_relation_codes(expected_rel)
            pred_dir, pred_dist = decode_relation_codes(predicted_rel)
            if exp_dir == pred_dir or exp_dist == pred_dist:
                matches += 0.5  # Partial credit for partial match

    return min(1.0, matches / total) if total > 0 else 0.0


def relations_consistency(pred_relations: Dict) -> float:
    """Check consistency of pairwise relations using spatial constraint solving.

    For each triple (A, B, C), set A at (0,0), add constraints A-B and A-C,
    then check if the implied B-C relation matches the given B-C relation.

    Args:
        pred_relations: Dict with pairwise relations like {"A|B": "(dir_code, dist_code)"}

    Returns:
        Float between 0.0 and 1.0 representing consistency (1.0 = fully consistent)
    """
    if not pred_relations or len(pred_relations) < 3:
        return 1.0  # Trivially consistent if too few relations

    # Extract unique object names
    all_names = set()
    for pair_key in pred_relations.keys():
        if '|' in pair_key:
            names = pair_key.split('|')
            if len(names) == 2:
                all_names.update(names)

    all_names = list(all_names)
    if len(all_names) < 3:
        return 1.0  # Need at least 3 objects for triangular consistency

    total_checks = 0
    consistent_checks = 0

    # Check consistency for each possible triple
    for i, name_a in enumerate(all_names):
        for j, name_b in enumerate(all_names[i+1:], i+1):
            for name_c in all_names[j+1:]:
                # Try both ab_key and ba_key directions
                ab_key = f"{name_a}|{name_b}"
                ba_key = f"{name_b}|{name_a}"
                ac_key = f"{name_a}|{name_c}"
                ca_key = f"{name_c}|{name_a}"
                bc_key = f"{name_b}|{name_c}"
                cb_key = f"{name_c}|{name_b}"

                # Try to find relations for this triangle
                ab_rel = pred_relations.get(ab_key) or pred_relations.get(ba_key)
                ac_rel = pred_relations.get(ac_key) or pred_relations.get(ca_key)
                bc_rel = pred_relations.get(bc_key) or pred_relations.get(cb_key)

                if ab_rel and ac_rel and bc_rel:
                    total_checks += 1

                    # Adjust relations based on actual key directions
                    # If we found ba_key instead of ab_key, invert the relation
                    if ab_key not in pred_relations and ba_key in pred_relations:
                        ab_rel = _invert_relation(ab_rel)

                    # If we found ca_key instead of ac_key, invert the relation
                    if ac_key not in pred_relations and ca_key in pred_relations:
                        ac_rel = _invert_relation(ac_rel)

                    # If we found cb_key instead of bc_key, invert the relation
                    if bc_key not in pred_relations and cb_key in pred_relations:
                        bc_rel = _invert_relation(bc_rel)

                    # Check consistency
                    if _check_triple_consistency(name_a, name_b, name_c, ab_rel, ac_rel, bc_rel):
                        consistent_checks += 1

    return consistent_checks / total_checks if total_checks > 0 else 1.0


def _check_triple_consistency(name_a: str, name_b: str, name_c: str,
                             ab_rel: str, ac_rel: str, bc_rel: str) -> bool:
    """Check if three pairwise relations form a consistent triangle.

    Uses spatial solver with AC and BC relations to get possible positions for A and B,
    then checks if any derived AB relation matches the given AB relation.
    """
    # Create spatial solver with the three objects
    solver = SpatialSolver([name_a, name_b, name_c], grid_size=10)

    # Set C at origin (easier to reason about AC and BC relations)
    solver.set_initial_position(name_c, (0, 0))

    # Parse relations to get codes
    ab_dir, ab_dist = decode_relation_codes(ab_rel)
    ac_dir, ac_dist = decode_relation_codes(ac_rel)
    bc_dir, bc_dist = decode_relation_codes(bc_rel)

    # Get representative positions and create constraints for AC and BC
    # AC relation: A relative to C
    ac_a_pos = _get_representative_position_from_codes(ac_dir, ac_dist, invert=False)
    # BC relation: B relative to C
    bc_b_pos = _get_representative_position_from_codes(bc_dir, bc_dist, invert=False)

    if ac_a_pos is None or bc_b_pos is None:
        return False

    # Create discrete relations from representative positions
    c_pos = (0, 0)
    ac_discrete_rel = PairwiseRelationshipDiscrete.relationship(
        ac_a_pos, c_pos, anchor_ori=(0, 1),
        bin_system=CardinalBinsAllo(), distance_bin_system=StandardDistanceBins()
    )
    bc_discrete_rel = PairwiseRelationshipDiscrete.relationship(
        bc_b_pos, c_pos, anchor_ori=(0, 1),
        bin_system=CardinalBinsAllo(), distance_bin_system=StandardDistanceBins()
    )

    # Add constraints: A relative to C, B relative to C
    from ...core.relationship import RelationTriple
    relation_triples = [
        RelationTriple(subject=name_a, anchor=name_c, relation=ac_discrete_rel, orientation=(0, 1)),
        RelationTriple(subject=name_b, anchor=name_c, relation=bc_discrete_rel, orientation=(0, 1))
    ]

    solver.add_observation(relation_triples)

    # Get possible positions from solver
    possible_positions = solver.get_possible_positions()
    a_positions = list(possible_positions.get(name_a, []))
    b_positions = list(possible_positions.get(name_b, []))

    if not a_positions or not b_positions:
        return False

    # For all possible (A, B) position pairs, check if any produces the predicted AB relation
    bin_system = CardinalBinsAllo()
    distance_bin_system = StandardDistanceBins()

    for a_pos in a_positions:
        for b_pos in b_positions:
            # Calculate A relative to B (since ab_rel describes "A|B" = A relative to B)
            actual_ab_rel = PairwiseRelationshipDiscrete.relationship(
                a_pos, b_pos, anchor_ori=(0, 1),
                bin_system=bin_system, distance_bin_system=distance_bin_system
            )
            actual_ab_dir = actual_ab_rel.direction.bin_label
            actual_ab_dist = actual_ab_rel.dist.bin_label

            # Convert to codes for comparison
            from ..relation_codes import to_code
            actual_ab_dir_code = to_code(actual_ab_dir)
            actual_ab_dist_code = to_code(actual_ab_dist)

            # Check if this matches the predicted AB relation
            if (actual_ab_dir_code.upper() == ab_dir.upper() and
                actual_ab_dist_code.lower() == ab_dist.lower()):
                return True

    return False

def _get_representative_position_from_codes(dir_code: str, dist_code: str, invert: bool = False) -> tuple:
    """Get a representative position from direction and distance codes.

    Args:
        dir_code: Direction code (N, S, E, W, etc.)
        dist_code: Distance code (near, mid, far, etc.)
        invert: If True, invert the direction

    Returns:
        A representative (x, y) position tuple, or None if invalid
    """
    from ...core.relationship import StandardDistanceBins

    # Invert direction if needed
    if invert:
        inversion_map = {
            'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E',
            'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'
        }
        dir_code = inversion_map.get(dir_code.upper(), dir_code)

    # Get a representative distance from the distance bin
    distance_bin_system = StandardDistanceBins()
    from ..relation_codes import _DIST_CODE_TO_LABEL
    dist_label = _DIST_CODE_TO_LABEL.get(dist_code.lower())

    if not dist_label:
        return None

    # Find the distance bin
    for i, label in enumerate(distance_bin_system.LABELS):
        if label.lower() == dist_label.lower():
            dist_min, dist_max = distance_bin_system.BINS[i]
            # Use middle of the range as representative distance
            rep_distance = (dist_min + dist_max) / 2
            break
    else:
        return None

    # Define direction unit vectors
    direction_vectors = {
        'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0),
        'NE': (1, 1), 'NW': (-1, 1), 'SE': (1, -1), 'SW': (-1, -1)
    }

    if dir_code.upper() not in direction_vectors:
        return None

    dx, dy = direction_vectors[dir_code.upper()]

    # Normalize for diagonal directions
    if dir_code.upper() in ['NE', 'NW', 'SE', 'SW']:
        length = (dx**2 + dy**2)**0.5
        dx, dy = dx/length, dy/length

    # Calculate representative position
    x = dx * rep_distance
    y = dy * rep_distance

    # Round to integer grid coordinates
    return (round(x), round(y))


def _invert_relation(relation_str: str) -> str:
    """Invert a relation string by reversing its direction.

    Args:
        relation_str: Relation string like "(W, near)"

    Returns:
        Inverted relation string like "(E, near)"
    """
    dir_code, dist_code = decode_relation_codes(relation_str)

    # Invert direction
    inversion_map = {
        'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E',
        'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'
    }
    inverted_dir = inversion_map.get(dir_code.upper(), dir_code)

    # Keep the same distance
    return f"({inverted_dir}, {dist_code})"

__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "rooms_vs_global_consistency",
    "map_vs_relations_consistency",
    "relations_consistency",
]



if __name__ == "__main__":
    import numpy as np
    from ...core.object import Object, Agent

    print("Testing consistency functions...")

    # Test 1: compare_on_common_subset
    print("\n1. Testing compare_on_common_subset:")
    try:
        # Create test BaseRooms with common objects
        obj1_a = Object(name="chair", pos=[1, 2])
        obj2_a = Object(name="table", pos=[3, 4])
        room_a = BaseRoom(objects=[obj1_a, obj2_a], name="room_a")

        obj1_b = Object(name="chair", pos=[1.1, 2.1])  # Slightly different position
        obj2_b = Object(name="table", pos=[3.2, 4.1])
        room_b = BaseRoom(objects=[obj1_b, obj2_b], name="room_b")

        metrics = compare_on_common_subset(room_a, room_b, allow_scale=False, pos_norm_L=None)
        print(f"Metrics: overall={metrics.overall:.3f}, pos={metrics.pos:.3f}, valid={metrics.valid}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: local_vs_global_consistency
    print("\n2. Testing local_vs_global_consistency:")
    try:
        # Create local and global rooms
        local_obj = Object(name="chair", pos=[0, 1])  # Relative to agent
        pred_local = BaseRoom(objects=[local_obj], name="local")

        global_obj = Object(name="chair", pos=[2, 3])  # Global position
        pred_global = BaseRoom(objects=[global_obj], name="global")

        agent = Agent(pos=[2, 2], ori=[0, 1])  # Agent at (2,2) facing north

        metrics = local_vs_global_consistency(pred_local, pred_global, agent, allow_scale=False, pos_norm_L=None)
        print(f"Metrics: overall={metrics.overall:.3f}, pos={metrics.pos:.3f}, valid={metrics.valid}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: map_vs_relations_consistency
    print("\n3. Testing map_vs_relations_consistency:")
    try:
        # Create BaseRoom with objects
        obj_a = Object(name="A", pos=[0, 0])
        obj_b = Object(name="B", pos=[1, 0])  # B is east of A
        obj_c = Object(name="C", pos=[0, 1])  # C is north of A
        pred_global = BaseRoom(objects=[obj_a, obj_b, obj_c], name="global")

        # Relations that should partially match the map
        # Note: "A|B" means A relative to B (A is relative to B)
        pred_relations = {
            "A|B": "(W, near)",     # A is west of B (since B is at [1,0] and A is at [0,0])
            "A|C": "(S, near)",     # A is south of C (since C is at [0,1] and A is at [0,0])
            "B|C": "(E, mid)"       # B is east of C (since B is at [1,0] and C is at [0,1])
        }

        score = map_vs_relations_consistency(pred_relations, pred_global)
        print(f"Map vs Relations consistency score: {score:.3f}")

        # Test with empty inputs
        score_empty = map_vs_relations_consistency({}, None)
        print(f"Empty inputs score: {score_empty:.3f}")

    except Exception as e:
        print(f"Error: {e}")

    # Test 4: relations_consistency
    print("\n4. Testing relations_consistency:")
    try:
        # Test with simple consistent relations (just 2 pairs)
        # Note: "A|B" means A relative to B
        simple_relations = {
            "A|B": "(W, near)",    # A is west of B
            "A|C": "(S, near)",    # A is south of C
        }

        # Test with triangle relations
        # Assume positions: A=[0,0], B=[1,0], C=[0,1]
        triangle_relations = {
            "A|B": "(W, near)",    # A is west of B (A=[0,0] relative to B=[1,0])
            "A|C": "(S, near)",    # A is south of C (A=[0,0] relative to C=[0,1])
            "B|C": "(SE, near)"     # B is east of C (B=[1,0] relative to C=[0,1])
        }

        # Test with inconsistent relations
        inconsistent_relations = {
            "A|B": "(W, near)",    # A is west of B (correct)
            "A|C": "(S, near)",    # A is south of C (correct)
            "B|C": "(W, far)"      # B is west of C (inconsistent - should be east)
        }

        score1 = relations_consistency(simple_relations)
        print(f"Simple relations (2 pairs) consistency: {score1:.3f}")

        score2 = relations_consistency(triangle_relations)
        print(f"Triangle relations consistency: {score2:.3f}")

        score3 = relations_consistency(inconsistent_relations)
        print(f"Inconsistent relations consistency: {score3:.3f}")

        # Test with empty/insufficient data
        score_empty = relations_consistency({})
        print(f"Empty relations consistency: {score_empty:.3f}")

        score_insufficient = relations_consistency({"A|B": "(E, near)"})
        print(f"Insufficient relations consistency: {score_insufficient:.3f}")

    except Exception as e:
        print(f"Error: {e}")

    print("\nConsistency function tests completed!")


