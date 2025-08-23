from ..core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, EgoFrontBins, StandardDistanceBins, CardinalBinsEgo, Dir, DirectionRel
from typing import Union
import math


def relationship_applies(obj1, obj2, relationship, anchor_ori: tuple = (0, 1)) -> bool:
    """Fast, allocation-free math check for Pairwise/Discrete/Proximity relationships."""
    p1 = getattr(obj1, 'pos', obj1)
    p2 = getattr(obj2, 'pos', obj2)

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x1 - x2, y1 - y2
    dsq = dx*dx + dy*dy

    ax, ay = float(anchor_ori[0]), float(anchor_ori[1])
    a_len = math.hypot(ax, ay) or 1.0
    axn, ayn = ax / a_len, ay / a_len

    if isinstance(relationship, PairwiseRelationshipDiscrete):
        # Distance bin check
        j = relationship.dist.bin_id
        if j == -1:
            return dsq <= 1e-6
        lo, hi = relationship.dist.bin_system.BINS[j]
        lo2, hi2 = float(lo)*float(lo), float(hi)*float(hi)
        if not (dsq > lo2 - 1e-6 and dsq <= hi2 + 1e-6):
            return False
        
        # Direction bin check
        bin_system = relationship.direction.bin_system
        # atan2(cross, dot) with normalized anchor; v length cancels out
        dot = axn*dx + ayn*dy
        cross = axn*dy - ayn*dx
        deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
        bid, _ = bin_system.bin(deg)
        return bid == relationship.direction.bin_id

    if isinstance(relationship, PairwiseRelationship):
        has_dir = relationship.direction is not None
        has_dist = relationship.dist is not None

        # Distance check when needed
        if has_dist:
            target_d = float(getattr(relationship.dist, 'value', 0.0))
            d = math.sqrt(dsq)
            if abs(d - target_d) > 1e-6:
                return False

        if has_dir:
            # compute dir pair relative to anchor ori using transform without object allocations
            h = Dir.SAME if abs(dx) < 1e-6 else (Dir.RIGHT if dx > 0 else Dir.LEFT)
            v = Dir.SAME if abs(dy) < 1e-6 else (Dir.FORWARD if dy > 0 else Dir.BACKWARD)
            ori_key = (int(anchor_ori[0]), int(anchor_ori[1]))
            t = DirectionRel.TRANSFORMS[ori_key]
            if t is None:
                th, tv = h, v
            else:
                th, tv = t[h], t[v]
            # swap rule from DirectionRel.transform
            if th in (Dir.FORWARD, Dir.BACKWARD) or tv in (Dir.RIGHT, Dir.LEFT):
                th, tv = tv, th
            rp = relationship.dir_pair
            if rp is None or (th != rp.horiz or tv != rp.vert):
                return False
            # degree check
            dot = axn*dx + ayn*dy
            cross = axn*dy - ayn*dx
            deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
            if abs(deg - float(relationship.degree)) > 1e-6:
                return False
        return True

    if isinstance(relationship, ProximityRelationship):
        th = float(getattr(relationship, 'PROXIMITY_THRESHOLD', 5.0))
        if dsq > th * th - 1e-6:
            return False
        # Must match the discrete pairwise inside the proximity relation
        return relationship_applies(p1, p2, relationship.pairwise_rel, anchor_ori)

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
            lo, hi = relationship.dist.bin_system.BINS[j]
            Rmin, Rmax = float(lo), float(hi)
    elif isinstance(relationship, PairwiseRelationship) and relationship.dist is not None:
        d = float(relationship.dist.value)
        Rmin, Rmax = max(0.0, d - 1e-6), d + 1e-6

    # If no distance bound, we do not generate (degree-only not supported)
    if Rmax is None:
        return out

    # x scan with y ranges from circle ring
    X0, X1 = max(xmin, int(math.ceil(ax - Rmax))), min(xmax, int(math.floor(ax + Rmax)))
    Rmax2, Rmin2 = float(Rmax * Rmax), float(max(Rmin, 0.0) * max(Rmin, 0.0))

    # Precompute for fast discrete checks
    axf, ayf = float(ax), float(ay)
    aox, aoy = float(anchor_ori[0]), float(anchor_ori[1])
    alen = math.hypot(aox, aoy) or 1.0
    aoxn, aoyn = aox/alen, aoy/alen

    is_disc = isinstance(relationship, PairwiseRelationshipDiscrete)
    disc_dir_bin = None
    disc_bin_system = None
    if is_disc:
        disc_dir_bin = relationship.direction.bin_id
        disc_bin_system = relationship.direction.bin_system or EgoFrontBins()

    for x in range(X0, X1 + 1):
        dx = float(x - ax)
        t2 = Rmax2 - dx*dx
        if t2 < 0: continue
        yspan = math.sqrt(t2) if t2 > 0.0 else 0.0
        y_top = int(math.floor(ay + yspan + 1e-6))
        y_bot = int(math.ceil(ay - yspan - 1e-6))
        if Rmin2 < 1e-6:
            for y in range(max(ymin, y_bot), min(ymax, y_top) + 1):
                if is_disc:
                    dy = float(y - ay)
                    dsq = dx*dx + dy*dy
                    if dsq <= 1e-6:
                        continue
                    # Direction bin fast check
                    dot = aoxn*dx + aoyn*dy
                    cross = aoxn*dy - aoyn*dx
                    deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
                    bid, _ = disc_bin_system.bin(deg)
                    if bid == disc_dir_bin:
                        out.add((x, y))
                else:
                    if relationship_applies((x, y), (ax, ay), relationship, anchor_ori):
                        out.add((x, y))
        else:
            t2in = Rmin2 - dx*dx
            if t2in > 1e-6:
                yin = math.sqrt(t2in)
                y_bot_in = int(math.ceil(ay - yin - 1e-6))
                y_top_in = int(math.floor(ay + yin + 1e-6))
            else:
                y_bot_in, y_top_in = ay, ay-1
            def _emit_range(y0: int, y1: int):
                if y0 > y1:
                    return
                y0, y1 = max(ymin, y0), min(ymax, y1)
                for y in range(y0, y1 + 1):
                    if x == ax and y == ay:
                        continue
                    if is_disc:
                        dy = float(y - ay)
                        dsq = dx*dx + dy*dy
                        # direction check only; distance ring already satisfied
                        dot = aoxn*dx + aoyn*dy
                        cross = aoxn*dy - aoyn*dx
                        deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
                        bid, _ = disc_bin_system.bin(deg)
                        if bid == disc_dir_bin:
                            out.add((x, y))
                    else:
                        if relationship_applies((x, y), (ax, ay), relationship, anchor_ori):
                            out.add((x, y))

            _emit_range(y_bot, y_bot_in - 1)
            _emit_range(y_top_in + 1, y_top)
    return out


if __name__ == "__main__":
    relationship = PairwiseRelationshipDiscrete.relationship((4, 6), (0, 0), anchor_ori=(1, 0), bin_system=CardinalBinsEgo(), distance_bin_system=StandardDistanceBins())
    points = generate_points_for_relationship((0, 0), relationship, (-20, 20), (-20, 20), (1, 0))
    for p in sorted(points):
        dist = math.hypot(p[0] - 0.0, p[1] - 0.0)
        print(f"Point {p}: distance = {dist:.2f}")