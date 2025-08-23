from ragen.env.spatial.Base.tos_base.core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, EgoFrontBins, StandardDistanceBins, CardinalBins
from typing import Union
import numpy as np


def relationship_applies(obj1, obj2, relationship, anchor_ori: tuple = (0, 1)) -> bool:
    """Check if a relationship applies. Supports Pairwise, PairwiseDiscrete, and ProximityRelationship."""
    p1 = getattr(obj1, 'pos', obj1)
    p2 = getattr(obj2, 'pos', obj2)

    if isinstance(relationship, PairwiseRelationshipDiscrete):
        # Use same bin and distance systems and perspective as the relationship
        bin_system = relationship.direction.bin_system or EgoFrontBins()
        distance_bins = relationship.dist.bin_system or StandardDistanceBins()
        perspective = getattr(relationship.direction, 'perspective', 'ego')
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(p1), tuple(p2), anchor_ori=anchor_ori,
            bin_system=bin_system, distance_bin_system=distance_bins,
            perspective=perspective
        )
        direction_match = rel.direction.bin_id == relationship.direction.bin_id
        distance_match = rel.dist.bin_id == relationship.dist.bin_id
        return direction_match and distance_match

    if isinstance(relationship, PairwiseRelationship):
        cur = PairwiseRelationship.relationship(tuple(p1), tuple(p2), anchor_ori=anchor_ori, full=True)
        # direction-only
        if relationship.dist is None and relationship.direction is not None:
            same_pair = cur.dir_pair == relationship.dir_pair
            deg_close = abs(cur.degree - relationship.degree) <= 1e-6
            return same_pair and deg_close
        # distance-only
        if relationship.direction is None and relationship.dist is not None:
            return abs(cur.distance_value - relationship.distance_value) <= 1e-6
        # full
        same_pair = cur.dir_pair == relationship.dir_pair
        deg_close = abs(cur.degree - relationship.degree) <= 1e-6
        dist_close = abs(cur.distance_value - relationship.distance_value) <= 1e-6
        return same_pair and deg_close and dist_close

    if isinstance(relationship, ProximityRelationship):
        cur = ProximityRelationship.from_positions(tuple(p1), tuple(p2), tuple(anchor_ori))
        if cur is None:
            return False
        # Check if the pairwise relationships match
        return relationship_applies(p1, p2, cur.pairwise_rel, anchor_ori)

    raise ValueError(f"Invalid relationship type: {type(relationship)}")



# ---- domain generator ----
def generate_points_for_relationship(
    anchor_pos: tuple,
    relationship: Union[PairwiseRelationship, PairwiseRelationshipDiscrete],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    anchor_ori: tuple[int, int] = (0, 1),
) -> set[tuple[int, int]]:
    """
    Generate integer points (x,y) within ranges that satisfy the relationship
    with the anchor_pos. The anchor_pos is treated as obj2 by default
    (i.e., we test relationship_applies(candidate, anchor_pos, ...)).

    Notes:
    - Only Pairwise/PairwiseDiscrete supported.
    - Handles distance-only or distance+degree. Degree-only not supported.
    """
    ax, ay = int(anchor_pos[0]), int(anchor_pos[1])
    xmin, xmax = int(x_range[0]), int(x_range[1])
    ymin, ymax = int(y_range[0]), int(y_range[1])

    out: set[tuple[int, int]] = set()

    # ---- Pairwise / PairwiseDiscrete ----
    # Determine distance window [Rmin, Rmax]
    Rmin, Rmax = 0.0, None
    if isinstance(relationship, PairwiseRelationshipDiscrete) and relationship.dist is not None:
        try:
            j = relationship.dist.bin_id
        except Exception:
            j = None
        if j == -1:
            return out
        if j is not None and relationship.dist.bin_system is not None:
            lo, hi, _ = relationship.dist.bin_system.BINS[j]
            Rmin, Rmax = float(lo), float(hi)
    elif isinstance(relationship, PairwiseRelationship) and relationship.dist is not None:
        d = float(relationship.dist.value)
        Rmin, Rmax = max(0.0, d - 1e-6), d + 1e-6

    # If no distance bound, we do not generate (degree-only not supported)
    if Rmax is None:
        return out

    # x scan with y ranges from circle ring
    X0, X1 = max(xmin, int(np.ceil(ax - Rmax))), min(xmax, int(np.floor(ax + Rmax)))
    Rmax2, Rmin2 = float(Rmax * Rmax), float(max(Rmin, 0.0) * max(Rmin, 0.0))

    for x in range(X0, X1 + 1):
        dx = float(x - ax)
        t2 = Rmax2 - dx*dx
        if t2 < 0: continue
        yspan = float(np.sqrt(max(t2, 0.0)))
        y_top = int(np.floor(ay + yspan + 1e-6))
        y_bot = int(np.ceil(ay - yspan - 1e-6))
        if Rmin2 < 1e-6:
            for y in range(max(ymin, y_bot), min(ymax, y_top) + 1):
                if relationship_applies((x, y), (ax, ay), relationship, anchor_ori):
                    out.add((x, y))
        else:
            t2in = Rmin2 - dx*dx
            if t2in > 1e-6:
                yin = float(np.sqrt(t2in))
                y_bot_in = int(np.ceil(ay - yin - 1e-6))
                y_top_in = int(np.floor(ay + yin + 1e-6))
            else:
                y_bot_in, y_top_in = ay, ay-1
            def _emit_range(y0: int, y1: int):
                if y0 > y1:
                    return
                y0, y1 = max(ymin, y0), min(ymax, y1)
                for y in range(y0, y1 + 1):
                    if x == ax and y == ay:
                        continue
                    if relationship_applies((x, y), (ax, ay), relationship, anchor_ori):
                        out.add((x, y))

            _emit_range(y_bot, y_bot_in - 1)
            _emit_range(y_top_in + 1, y_top)
    return out


if __name__ == "__main__":
    relationship = PairwiseRelationshipDiscrete.relationship((4, 6), (0, 0), anchor_ori=(1, 0), bin_system=CardinalBins(), distance_bin_system=StandardDistanceBins(), perspective='ego')
    points = generate_points_for_relationship((0, 0), relationship, (-20, 20), (-20, 20), (1, 0))
    # print(points)
    for p in sorted(points):
        dist = np.linalg.norm(np.array(p) - np.array((0, 0)))
        print(f"Point {p}: distance = {dist:.2f}")