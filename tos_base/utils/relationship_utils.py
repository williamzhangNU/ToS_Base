from ..core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, EgoFrontBins, StandardDistanceBins, CardinalBinsEgo
from typing import Union
import math
import numpy as np
import matplotlib.pyplot as plt


def relationship_applies(obj1, obj2, relationship, anchor_ori: tuple = (0, 1)) -> bool:
    """Check if relationship applies to obj1 and obj2 from anchor's perspective."""
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
        # Distance bin check (open interval)
        j = relationship.dist.bin_id
        lo, hi = relationship.dist.bin_system.BINS[j]
        d = math.sqrt(dsq)
        if not (d > float(lo) and d < float(hi)):
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
            # Only degree comparison is needed now
            dot = axn*dx + ayn*dy
            cross = axn*dy - ayn*dx
            deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
            if abs(deg - float(relationship.degree)) > 1e-6:
                return False
        return True

    if isinstance(relationship, ProximityRelationship):
        th = float(getattr(relationship, 'PROXIMITY_THRESHOLD', 5.0))
        if not (dsq < th * th):
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

    TODO debug
    """
    ax, ay = int(anchor_pos[0]), int(anchor_pos[1])
    xmin, xmax = int(x_range[0]), int(x_range[1])
    ymin, ymax = int(y_range[0]), int(y_range[1])

    out: set[tuple[int, int]] = set()

    # ---- Pairwise / PairwiseDiscrete ----
    # Determine distance window [Rmin, Rmax]
    Rmin, Rmax = 0.0, None
    if isinstance(relationship, PairwiseRelationshipDiscrete) and relationship.dist is not None:
        j = relationship.dist.bin_id
        if j is not None and relationship.dist.bin_system is not None:
            lo, hi = relationship.dist.bin_system.BINS[j]
            # same-distance bin detected by j==0
            if j == 0:
                return out
            Rmin, Rmax = float(lo), float(hi)
    elif isinstance(relationship, PairwiseRelationship) and relationship.dist is not None:
        d = float(relationship.dist.value)
        Rmin, Rmax = max(0.0, d - 1e-6), d + 1e-6

    # If no distance bound, we do not generate (degree-only not supported)
    if Rmax is None:
        return out

    # x scan with y ranges from circle ring
    X0 = max(xmin, int(math.ceil(ax - Rmax + 1e-9)))
    X1 = min(xmax, int(math.floor(ax + Rmax - 1e-9)))
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
        y_top = int(math.floor(ay + yspan - 1e-9))
        y_bot = int(math.ceil(ay - yspan + 1e-9))
        for y in range(max(ymin, y_bot), min(ymax, y_top) + 1):
            if x == ax and y == ay:
                continue
            dy = float(y - ay)
            dsq = dx*dx + dy*dy
            if not (dsq > Rmin2 and dsq < Rmax2):
                continue
            if is_disc:
                dot = aoxn*dx + aoyn*dy
                cross = aoxn*dy - aoyn*dx
                deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
                bid, _ = disc_bin_system.bin(deg)
                if bid == disc_dir_bin:
                    out.add((x, y))
            else:
                if relationship_applies((x, y), (ax, ay), relationship, anchor_ori):
                    out.add((x, y))
    return out



# ------------ rose glyph for paper plot ------------

def rose_glyph_pretty(
    pairs,                      # list of (sector_id, ring_id)
    color="#E6A23C",
    ring_colors=("#E8F3FF", "#EAF7F0", "#FFF3E0", "#F3E8FF"),
    tick_colors=("#4C78A8","#8E9ED6","#74C0E3","#BDE0FE","#A8DADC","#95D5B2","#FFD166","#F4A261"),
    gap=0.92,
    n_dirs=8,
    n_rings=4,
    figsize=(2.4, 2.4),
    ring_heights=(0.4, 0.25, 0.25, 0.25),
    bg_band_ratio=1,          # 0..1, portion of eacF4A261h ring colored near the outer edge
    sector_highlight_color="#FDE68A",
    sector_highlight_alpha=0.9
):
    """Rose glyph with (sector, ring) fills; background rings colored on outer band, and selected sectors highlighted."""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_rlim(0, 1)

    # radial boundaries
    rh = np.array(ring_heights[:n_rings], dtype=float); rh /= rh.sum()
    rb = np.concatenate([[0.0], np.cumsum(rh)])

    sector_angle = 2*np.pi / n_dirs
    width = sector_angle * gap
    sectors = {int(s) % n_dirs for s, _ in pairs}

    # background rings: only outer band portion
    for i, c in enumerate(ring_colors[:n_rings]):
        h = rb[i+1] - rb[i]
        band_h = h * max(0.0, min(1.0, bg_band_ratio))
        bottom = rb[i+1] - band_h
        ax.bar(0, band_h, width=2*np.pi, bottom=bottom, color=c, alpha=0.5, linewidth=0)

    # sector highlights (any sector appearing in `pairs`)
    for s in sectors:
        theta = s * sector_angle
        ax.bar(theta, 1.0, width=width, bottom=0.0, color=sector_highlight_color,
               linewidth=0, alpha=sector_highlight_alpha, align='center')

    # target (sector, ring) wedges
    theta_map = {s: s * sector_angle for s in sectors}
    for s, r in pairs:
        s = int(s) % n_dirs; r = int(r)
        if 0 <= r < n_rings:
            theta = theta_map.get(s, s * sector_angle)
            ax.bar(theta, rb[r+1]-rb[r], width=width, bottom=rb[r],
                   align='center', linewidth=0, color=color)

    # ring separators
    th = np.linspace(0, 2*np.pi, 361)
    for r in rb[1:-1]: ax.plot(th, np.full_like(th, r), lw=1.1, color="white")

    # ticks
    centers = np.arange(n_dirs) * sector_angle
    for i, th_c in enumerate(centers):
        ax.plot([th_c, th_c], [1.02, 1.08], lw=2.4,
                color=tick_colors[i % len(tick_colors)], solid_capstyle='round')

    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_rlim(0, 1.10)
    return fig, ax



if __name__ == "__main__":
    # relationship = PairwiseRelationshipDiscrete.relationship((4, 6), (0, 0), anchor_ori=(1, 0), bin_system=CardinalBinsEgo(), distance_bin_system=StandardDistanceBins())
    # points = generate_points_for_relationship((0, 0), relationship, (-20, 20), (-20, 20), (1, 0))
    # for p in sorted(points):
    #     dist = math.hypot(p[0] - 0.0, p[1] - 0.0)
    #     print(f"Point {p}: distance = {dist:.2f}")

    # 扇区: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    # 环:   near=0, mid=1, slightly far=2, far=3
    # ax = rose_glyph_pretty(pairs=[(7, 1), (6, 1), (7, 2), (6, 2)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(7, 2), (0, 1), (7, 1)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(6, 1), (5, 2), (5, 1), (6, 2)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(6, 3), (7, 2), (6, 2), (7, 3)], color="#E59E1B")
    ax = rose_glyph_pretty(pairs=[(3, 2)], color="#E59E1B")

    plt.savefig("rose_glyph_pretty.png", transparent=True)