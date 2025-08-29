import numpy as np
from typing import Tuple, List
import random

def generate_rooms(n: int, level: int, main: int = None, seed: int = None, debug: bool = False,
                  topology: str = "tree") -> np.ndarray:
    """
    Function to generate room layout

    Args:
        n: Grid size (n x n)
        level: Complexity level, level=0 means 1 room, level=1 means 2 rooms, and so on
        main: Main room size, if specified the first room will be main×main size
        seed: Random seed
        debug: Debug mode flag
        topology: Room connection topology, options:
            - "tree": Minimum spanning tree (default, no cycles)
            - "line": Linear connection (1→2→3→4)
            - "star": Star-like connection (1→2, 1→3, 1→4)

    Returns:
        n x n numpy array where:
        - 1 to level+1: Room ID
        - 0: Wall
        - -1: Impassable area (outside rooms)
        - 100: North-south door
        - 101: East-west door
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_rooms = level + 1

    # Initialize grid, all as impassable area
    grid = np.full((n, n), -1, dtype=int)

    if num_rooms == 1:
        return _generate_single_room(grid, n, main)

    # Multiple attempts to generate valid room layout
    max_attempts = 100
    for attempt in range(max_attempts):
        # Reset grid
        grid = np.full((n, n), -1, dtype=int)

        # Generate room layout based on topology
        if topology == "line":
            rooms = _generate_line_room_layout(n, num_rooms, main)
        elif topology == "star":
            rooms = _generate_star_room_layout(n, num_rooms, main)
        elif topology == "tree":
            rooms = _generate_tree_room_layout(n, num_rooms, main)
        else:
            rooms = _generate_room_layout(n, num_rooms, main)
        if not rooms or len(rooms) != num_rooms:
            continue

        # Generate room connections based on topology
        connections = _generate_connections(rooms, topology)

        # Place rooms in grid
        for i, room in enumerate(rooms):
            room_id = i + 1
            x1, y1, x2, y2 = room
            grid[y1:y2+1, x1:x2+1] = room_id

        # Add walls
        _add_walls_around_rooms(grid, rooms)

        # Add doors
        doors_added = _add_doors_between_rooms(grid, rooms, connections)

        # Check if all doors were successfully added and connected
        if len(doors_added) == len(connections) and _verify_connectivity(grid, num_rooms):
            # Final check: ensure no rooms reach the boundary
            if _verify_no_rooms_at_boundary(grid, num_rooms):
                return grid

    # If multiple attempts fail, return single room
    return _generate_single_room(grid, n, main)


def generate_rooms_auto(level: int, main: int = None, seed: int = None, debug: bool = False,
                         topology: str = "tree", canvas_n: int = None) -> np.ndarray:
    """
    自动生成房间布局：不需显式传入 n，返回尽量小的 m×n 掩码(mask)。

    约束：
    - 最外层四条边都不应全部为 -1（裁剪到包含非 -1 的最小包围矩形即可满足）
    - 墙为单层（沿用现有围墙逻辑，裁剪不会增厚墙）

    Args:
        level: 复杂度（房间数为 level+1）
        main: 主房间边长（可选）
        seed: 随机种子（可选）
        debug: 调试标志（可选）
        topology: 拓扑（tree/line/star）
        canvas_n: 内部初始画布尺寸（可选）；若不传，根据 level/main 估算

    Returns:
        最小化裁剪后的 m×n numpy 数组
    """
    # 估算一个安全的初始画布大小，用于调用现有 generate_rooms；随后裁剪为最小尺寸
    num_rooms = level + 1

    def _suggest_canvas_size() -> int:
        # 基于主房间和房间数量的粗略估计，宁宽勿窄，后续会裁剪
        base_room = max(6, (main or 6))

        if topology == "line":
            # 线性布局：房间排成一行，需要的空间相对可预测
            est = base_room + (num_rooms - 1) * 7 + 6  # 每个房间+墙+门
        elif topology == "star":
            # 星形布局：主房间居中，其他房间围绕，空间需求适中
            est = base_room * 3 + 8  # 主房间+四周房间+墙
        else:  # tree topology
            # 树形布局：房间随机分布，需要更大空间确保能找到合适位置和连接
            # 增加更多空间以提高成功率
            est = max(30, base_room * 2 + num_rooms * 8 + 10)

        return max(15, min(100, est))  # 提高最小值和上限

    n = canvas_n or _suggest_canvas_size()

    # 先用固定画布生成
    full_grid = generate_rooms(n, level, main=main, seed=seed, debug=debug, topology=topology)

    # 再裁剪至包含所有非 -1 的最小包围矩形
    cropped = _crop_to_minimal_mask(full_grid)

    # 兜底：若异常导致全为 -1，则直接返回 full_grid
    if cropped is None:
        return full_grid

    return cropped


def _crop_to_minimal_mask(grid: np.ndarray) -> np.ndarray:
    """
    将网格裁剪到最小包围矩形：该矩形包含所有非 -1 的单元（房间/墙/门）。
    - 裁剪后四条边一定至少含有一个非 -1 值（满足“边不全为 -1”）。
    - 不改变内部墙厚度（保持单层）。

    Returns:
        裁剪后的网格；若网格全为 -1，返回 None
    """
    h, w = grid.shape

    # 找到所有非 -1 的行与列
    rows_with_content = [i for i in range(h) if np.any(grid[i, :] != -1)]
    cols_with_content = [j for j in range(w) if np.any(grid[:, j] != -1)]

    if not rows_with_content or not cols_with_content:
        return None

    r1, r2 = rows_with_content[0], rows_with_content[-1]
    c1, c2 = cols_with_content[0], cols_with_content[-1]

    return grid[r1:r2+1, c1:c2+1]


def _generate_single_room(grid: np.ndarray, n: int, main: int = None) -> np.ndarray:
    """Generate single room, occupying center area"""
    # Calculate room size, ensure space for surrounding walls
    if main is not None:
        # If main parameter is specified, use main×main as room size
        room_size = main
        if room_size > n - 4:  # Ensure room is not too large
            room_size = n - 4
        if room_size < 4:  # Ensure room is not too small
            room_size = 4
    else:
        # Default room size calculation
        min_room_size = 4
        max_room_size = n - 4  # Leave space for walls
        room_size = max(min_room_size, max_room_size)

    # Calculate room position (centered)
    start = (n - room_size) // 2
    end = start + room_size

    # Ensure room doesn't reach boundary, must have space for surrounding walls
    if end >= n - 1:  # Leave at least 1 cell for wall
        end = n - 2
        start = end - room_size + 1
    if start < 1:  # Leave at least 1 cell for wall
        start = 1
        end = start + room_size - 1

    # Place room
    grid[start:end+1, start:end+1] = 1

    # Add surrounding walls
    wall_start_x = max(0, start - 1)
    wall_end_x = min(n - 1, end + 1)
    wall_start_y = max(0, start - 1)
    wall_end_y = min(n - 1, end + 1)

    # First set wall area
    grid[wall_start_y:wall_end_y+1, wall_start_x:wall_end_x+1] = 0
    # Then set room area
    grid[start:end+1, start:end+1] = 1

    return grid

def _generate_line_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """
    为 line 拓扑生成房间布局：
    - 连接关系为链式：1→2→3→4→5
    - 房间物理位置不必排成直线，可以更灵活（如L形、Z形等）
    - 采用增量式放置：每个新房间只需与前一个房间相邻即可
    """
    if num_rooms <= 0:
        return []

    if num_rooms == 1:
        # Single room, use regular layout
        return _generate_room_layout(n, num_rooms, main)

    rooms: List[Tuple[int, int, int, int]] = []

    # 主房间尺寸 - 固定尺寸，不随seed变化
    if main is not None:
        main_size = max(4, min(main, n - 4))
    else:
        main_size = max(6, min(8, n // 4))

    # 其他房间尺寸 - 增加随机变化，但要小于主房间
    base_other_size = max(4, min(main_size - 1, n // 6))
    other_size_variation = np.random.randint(-2, 4)
    other_size = max(4, min(base_other_size + other_size_variation, main_size - 1, n // 6))

    # 第一个房间（主房间）放在中心附近 - 主房间始终是正方形
    mw = mh = main_size

    # 主房间位置增加随机偏移
    center_offset_x = np.random.randint(-2, 3)
    center_offset_y = np.random.randint(-2, 3)
    cx = max(mw//2 + 2, min(n - mw//2 - 2, n // 2 + center_offset_x))
    cy = max(mh//2 + 2, min(n - mh//2 - 2, n // 2 + center_offset_y))

    x1 = max(1, cx - mw // 2)
    y1 = max(1, cy - mh // 2)
    x2 = min(n - 2, x1 + mw - 1)
    y2 = min(n - 2, y1 + mh - 1)
    # 调整位置确保尺寸正确
    x1 = x2 - mw + 1
    y1 = y2 - mh + 1

    main_room = (x1, y1, x2, y2)
    rooms.append(main_room)

    # 链式放置其他房间：每个新房间只与前一个房间相邻
    for room_idx in range(1, num_rooms):
        placed = False
        attempts = 0
        max_attempts = 50

        # 新房间只与前一个房间（rooms[room_idx-1]）相邻
        prev_room = rooms[room_idx-1]
        px1, py1, px2, py2 = prev_room

        while not placed and attempts < max_attempts:
            # 尝试在前一个房间的四个方向放置新房间
            directions = ["left", "right", "top", "bottom"]
            np.random.shuffle(directions)

            for direction in directions:
                # 新房间尺寸（增加更大的变化和矩形可能性），但要小于主房间
                size_variation = np.random.randint(-1, 4)
                base_w = max(4, min(other_size + size_variation, main_size - 1))
                base_h = max(4, min(other_size + size_variation, main_size - 1))

                # 70% 概率生成矩形房间（line topology 更倾向于矩形）
                if np.random.random() < 0.7:
                    if np.random.random() < 0.5:
                        # 宽房间
                        rw = min(base_w + np.random.randint(1, 4), main_size - 1)
                        rh = max(4, base_h - np.random.randint(0, 2))
                    else:
                        # 高房间
                        rw = max(4, base_w - np.random.randint(0, 2))
                        rh = min(base_h + np.random.randint(1, 4), main_size - 1)
                else:
                    # 方形房间
                    rw = rh = base_w

                if direction == "left":
                    # 新房间在前一个房间左侧
                    rx2 = px1 - 2  # 留1格放墙/门
                    rx1 = rx2 - rw + 1
                    # y位置：与前一个房间有重叠以便放门
                    ry1 = py1 + np.random.randint(0, max(1, (py2 - py1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "right":
                    # 新房间在前一个房间右侧
                    rx1 = px2 + 2
                    rx2 = rx1 + rw - 1
                    ry1 = py1 + np.random.randint(0, max(1, (py2 - py1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "top":
                    # 新房间在前一个房间上方
                    ry2 = py1 - 2
                    ry1 = ry2 - rh + 1
                    rx1 = px1 + np.random.randint(0, max(1, (px2 - px1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                else:  # bottom
                    # 新房间在前一个房间下方
                    ry1 = py2 + 2
                    ry2 = ry1 + rh - 1
                    rx1 = px1 + np.random.randint(0, max(1, (px2 - px1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                new_room = (rx1, ry1, rx2, ry2)

                # 检查新房间是否在边界内且不与现有房间重叠
                if (rx1 >= 1 and ry1 >= 1 and rx2 <= n - 2 and ry2 <= n - 2 and
                    not _rooms_overlap_with_walls(new_room, rooms)):
                    rooms.append(new_room)
                    placed = True
                    break

            attempts += 1

        if not placed:
            # 如果无法放置，返回已放置的房间（可能少于要求数量）
            break

    return rooms


def _generate_star_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """
    为 star 拓扑生成布局：
    - 第 1 个房间为主房间，放在中心；
    - 其余房间分别贴近主房间的四侧（左/右/上/下），与主房间之间保留 1 格（用于墙/门）。
    - 确保与主房间在相邻方向存在重叠区间，以便放置门（_find_door_between_rooms 需要 overlap）。
    若放置失败则返回空列表让上层重试。
    """
    if num_rooms <= 0:
        return []

    rooms: List[Tuple[int, int, int, int]] = []

    # 主房间尺寸 - 固定尺寸，不随seed变化
    if main is not None:
        main_size = max(4, min(main, n - 4))
    else:
        main_size = max(6, min(10, n // 3))

    # 主房间放在中心，留边界 1 供外墙 - 主房间始终是正方形
    mw = mh = main_size

    # 主房间位置增加小幅随机偏移
    center_offset_x = np.random.randint(-2, 3)
    center_offset_y = np.random.randint(-2, 3)
    cx = max(mw//2 + 2, min(n - mw//2 - 2, n // 2 + center_offset_x))
    cy = max(mh//2 + 2, min(n - mh//2 - 2, n // 2 + center_offset_y))

    x1 = max(1, cx - mw // 2)
    y1 = max(1, cy - mh // 2)
    x2 = min(n - 2, x1 + mw - 1)
    y2 = min(n - 2, y1 + mh - 1)
    # 若被边界截断导致尺寸变化，修正左上
    x1 = x2 - mw + 1
    y1 = y2 - mh + 1

    main_room = (x1, y1, x2, y2)
    rooms.append(main_room)

    # 其他房间尺寸基准 - 增加随机变化，但要小于主房间
    base_other = max(4, min(main_size - 1, max(4, n // (num_rooms + 2))))
    other_size_variation = np.random.randint(-2, 4)
    base_other = max(4, min(base_other + other_size_variation, main_size - 1, n // (num_rooms + 2)))

    # 侧向按顺序循环：左、右、上、下
    sides = ["left", "right", "top", "bottom"]

    def fits(room: Tuple[int,int,int,int]) -> bool:
        x1, y1, x2, y2 = room
        if x1 < 1 or y1 < 1 or x2 > n - 2 or y2 > n - 2:
            return False
        return not _rooms_overlap_with_walls(room, rooms)

    # 尝试为其余 num_rooms-1 个房间在主房间四周找位置
    idx = 0
    attempts_per_room = 20
    for i in range(1, num_rooms):
        side = sides[idx % len(sides)]
        idx += 1

        # 为竖直侧（left/right）优先让高度不超过主房间高度，以保证 overlap；
        # 为水平侧（top/bottom）优先让宽度不超过主房间宽度。
        # 增加房间尺寸的随机变化和矩形可能性
        if side in ("left", "right"):
            base_h = min(base_other, (y2 - y1 + 1))
            base_w = base_other
        else:
            base_w = min(base_other, (x2 - x1 + 1))
            base_h = base_other

        # 增加矩形变化，但要小于主房间
        if np.random.random() < 0.6:  # 60% 概率生成矩形
            if side in ("left", "right"):
                # 竖直侧：可以调整宽度
                w = max(4, min(base_w + np.random.randint(-1, 3), main_size - 1))
                h = base_h
            else:
                # 水平侧：可以调整高度
                w = base_w
                h = max(4, min(base_h + np.random.randint(-1, 3), main_size - 1))
        else:
            w, h = base_w, base_h

        placed = False
        # 如果默认尺寸放不下，逐步缩小
        for size_shrink in range(0, 4):
            ww = max(4, min(w - size_shrink, main_size - 1))
            hh = max(4, min(h - size_shrink, main_size - 1))

            # 计算初始位置（与主房间中心对齐）并允许沿着接触边滑动
            if side == "left":
                rx2 = x1 - 2  # 与主房间左侧相距 1 格
                rx1 = rx2 - ww + 1
                # y 居中对齐
                ry1_init = y1 + ((y2 - y1 + 1) - hh) // 2
                # 滑动范围：让其与主房间垂直方向保持 overlap
                y_min = y1
                y_max = y2 - hh + 1
                candidates = list(range(ry1_init, ry1_init + 1)) + list(range(y_min, y_max + 1))
                for ry1 in candidates:
                    ry2 = ry1 + hh - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            elif side == "right":
                rx1 = x2 + 2
                rx2 = rx1 + ww - 1
                ry1_init = y1 + ((y2 - y1 + 1) - hh) // 2
                y_min = y1
                y_max = y2 - hh + 1
                candidates = list(range(ry1_init, ry1_init + 1)) + list(range(y_min, y_max + 1))
                for ry1 in candidates:
                    ry2 = ry1 + hh - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            elif side == "top":
                ry2 = y1 - 2
                ry1 = ry2 - hh + 1
                rx1_init = x1 + ((x2 - x1 + 1) - ww) // 2
                x_min = x1
                x_max = x2 - ww + 1
                candidates = list(range(rx1_init, rx1_init + 1)) + list(range(x_min, x_max + 1))
                for rx1 in candidates:
                    rx2 = rx1 + ww - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            elif side == "bottom":
                ry1 = y2 + 2
                ry2 = ry1 + hh - 1
                rx1_init = x1 + ((x2 - x1 + 1) - ww) // 2
                x_min = x1
                x_max = x2 - ww + 1
                candidates = list(range(rx1_init, rx1_init + 1)) + list(range(x_min, x_max + 1))
                for rx1 in candidates:
                    rx2 = rx1 + ww - 1
                    room = (rx1, ry1, rx2, ry2)
                    if fits(room):
                        rooms.append(room)
                        placed = True
                        break

            if placed:
                break

        if not placed:
            # 放置失败，返回空列表让上层重试
            return []

    return rooms


def _generate_tree_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """
    为 tree 拓扑生成房间布局：
    - 采用增量式放置：先放主房间，然后逐个放置其他房间，确保每个新房间与已有房间相邻
    - 这样生成的布局天然适合树状连接，因为每个房间都能与至少一个已有房间相邻
    """
    if num_rooms <= 0:
        return []

    rooms: List[Tuple[int, int, int, int]] = []

    # 主房间尺寸 - 固定尺寸，不随seed变化
    if main is not None:
        main_size = max(4, min(main, n - 4))
    else:
        main_size = max(6, min(8, n // 4))

    # 其他房间尺寸 - 增加更大的随机变化范围，但要小于主房间
    base_other_size = max(4, min(main_size - 1, n // 6))
    other_size_variation = np.random.randint(-2, 4)
    other_size = max(4, min(base_other_size + other_size_variation, main_size - 1, n // 6))

    # 第一个房间（主房间）放在中心附近 - 主房间始终是正方形
    mw = mh = main_size

    # 主房间位置增加随机偏移
    center_offset_x = np.random.randint(-3, 4)
    center_offset_y = np.random.randint(-3, 4)
    cx = max(mw//2 + 2, min(n - mw//2 - 2, n // 2 + center_offset_x))
    cy = max(mh//2 + 2, min(n - mh//2 - 2, n // 2 + center_offset_y))

    x1 = max(1, cx - mw // 2)
    y1 = max(1, cy - mh // 2)
    x2 = min(n - 2, x1 + mw - 1)
    y2 = min(n - 2, y1 + mh - 1)
    # 调整位置确保尺寸正确
    x1 = x2 - mw + 1
    y1 = y2 - mh + 1

    main_room = (x1, y1, x2, y2)
    rooms.append(main_room)

    # 增量式放置其他房间
    for room_idx in range(1, num_rooms):
        placed = False
        attempts = 0
        max_attempts = 50

        while not placed and attempts < max_attempts:
            # 随机选择一个已有房间作为"邻居"
            neighbor_idx = np.random.randint(0, len(rooms))
            neighbor = rooms[neighbor_idx]
            nx1, ny1, nx2, ny2 = neighbor

            # 尝试在邻居房间的四个方向放置新房间
            directions = ["left", "right", "top", "bottom"]
            np.random.shuffle(directions)

            for direction in directions:
                # 新房间尺寸（增加更大的变化和矩形可能性），但要小于主房间
                size_variation = np.random.randint(-1, 4)  # 扩大变化范围
                base_w = max(4, min(other_size + size_variation, main_size - 1))
                base_h = max(4, min(other_size + size_variation, main_size - 1))

                # 60% 概率生成矩形房间
                if np.random.random() < 0.6:
                    if np.random.random() < 0.5:
                        # 宽房间
                        rw = min(base_w + np.random.randint(1, 4), main_size - 1)
                        rh = max(4, base_h - np.random.randint(0, 2))
                    else:
                        # 高房间
                        rw = max(4, base_w - np.random.randint(0, 2))
                        rh = min(base_h + np.random.randint(1, 4), main_size - 1)
                else:
                    # 方形房间
                    rw = rh = base_w

                if direction == "left":
                    # 新房间在邻居左侧
                    rx2 = nx1 - 2  # 留1格放墙/门
                    rx1 = rx2 - rw + 1
                    # y位置：与邻居有重叠以便放门
                    ry1 = ny1 + np.random.randint(0, max(1, (ny2 - ny1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "right":
                    # 新房间在邻居右侧
                    rx1 = nx2 + 2
                    rx2 = rx1 + rw - 1
                    ry1 = ny1 + np.random.randint(0, max(1, (ny2 - ny1 + 1) - rh + 1))
                    ry2 = ry1 + rh - 1

                elif direction == "top":
                    # 新房间在邻居上方
                    ry2 = ny1 - 2
                    ry1 = ry2 - rh + 1
                    rx1 = nx1 + np.random.randint(0, max(1, (nx2 - nx1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                else:  # bottom
                    # 新房间在邻居下方
                    ry1 = ny2 + 2
                    ry2 = ry1 + rh - 1
                    rx1 = nx1 + np.random.randint(0, max(1, (nx2 - nx1 + 1) - rw + 1))
                    rx2 = rx1 + rw - 1

                new_room = (rx1, ry1, rx2, ry2)

                # 检查新房间是否在边界内且不与现有房间重叠
                if (rx1 >= 1 and ry1 >= 1 and rx2 <= n - 2 and ry2 <= n - 2 and
                    not _rooms_overlap_with_walls(new_room, rooms)):
                    rooms.append(new_room)
                    placed = True
                    break

            attempts += 1

        if not placed:
            # 如果无法放置，返回已放置的房间（可能少于要求数量）
            break

    return rooms

def _generate_room_layout(n: int, num_rooms: int, main: int = None) -> List[Tuple[int, int, int, int]]:
    """Generate room layout, return list of room coordinates (x1, y1, x2, y2)"""
    rooms = []

    # Adjust room size based on number of rooms, increase differentiation
    if num_rooms == 1:
        # Single room, can be larger
        min_size = max(4, n // 4)
        max_size = max(min_size, n - 4)
    else:
        # Multiple rooms, increase size differentiation
        min_size = 4
        # Dynamically adjust maximum size based on grid size and number of rooms
        if n >= 20:
            max_size = max(min_size, min(10, n // max(2, num_rooms - 1)))
        else:
            max_size = max(min_size, min(8, n // (num_rooms + 1)))

    max_attempts = 1000

    for i in range(num_rooms):
        attempts = 0
        placed = False

        while attempts < max_attempts and not placed:
            # Generate different sizes for each room, increase differentiation
            if i == 0 and main is not None:
                # First room uses specified main×main size
                width = main
                height = main
                # Ensure main room is not too large (based on grid size, not max_size)
                max_possible_size = n - 4  # Leave space for walls
                if width > max_possible_size:
                    width = max_possible_size
                if height > max_possible_size:
                    height = max_possible_size
                # Ensure main room is not too small
                if width < min_size:
                    width = min_size
                if height < min_size:
                    height = min_size
            elif num_rooms > 1:
                # Use different size preferences for different rooms, increase larger differences
                if i == 0:
                    # First room is larger (if main is not specified)
                    room_min = min_size
                    room_max = max_size
                elif i == 1:
                    # Second room is medium size
                    room_min = min_size
                    room_max = max(min_size, max_size - random.randint(1, 2))
                elif i == 2:
                    # Third room is smaller
                    room_min = min_size
                    room_max = max(min_size, max_size - random.randint(2, 3))
                else:
                    # Other rooms have random size variation
                    size_variation = random.choice([0, 1, 2, 3])
                    room_min = min_size
                    room_max = max(min_size, max_size - size_variation)

                # Generate room size, width and height can be different
                width = random.randint(room_min, room_max)
                height = random.randint(room_min, room_max)

                # Increase probability of rectangular rooms with larger differences
                if random.random() < 0.6:  # 60% probability to generate rectangle
                    if random.random() < 0.5:
                        # Increase width
                        width = min(max_size, width + random.randint(1, 3))
                    else:
                        # Increase height
                        height = min(max_size, height + random.randint(1, 3))
            else:
                # Single room case
                room_min = min_size
                room_max = max_size
                width = random.randint(room_min, room_max)
                height = random.randint(room_min, room_max)

            # Ensure room doesn't reach boundary, must have space for surrounding walls
            max_x = n - width - 1  # Leave at least 1 cell for wall
            max_y = n - height - 1  # Leave at least 1 cell for wall

            if max_x < 1 or max_y < 1:
                # Not enough space, shrink room
                width = max(min_size, n - 2)  # Leave 1 cell on each side
                height = max(min_size, n - 2)  # Leave 1 cell on each side
                max_x = n - width - 1
                max_y = n - height - 1

                if max_x < 1 or max_y < 1:
                    break

            # Randomly generate room position (starting from 1, ensure space for surrounding walls)
            x1 = random.randint(1, max_x)
            y1 = random.randint(1, max_y)
            x2 = x1 + width - 1
            y2 = y1 + height - 1

            new_room = (x1, y1, x2, y2)

            # Check if it overlaps with existing rooms
            if not _rooms_overlap_with_walls(new_room, rooms):
                rooms.append(new_room)
                placed = True

            attempts += 1

        if not placed:
            # Try to force place a smaller room
            forced_room = _force_place_small_room(n, rooms)
            if forced_room:
                rooms.append(forced_room)
            else:
                # If even forced placement fails, stop generating more rooms
                break

    return rooms

def _rooms_overlap_with_walls(new_room: Tuple[int, int, int, int],
                             existing_rooms: List[Tuple[int, int, int, int]]) -> bool:
    """Check if rooms overlap (including space for walls and doors)"""
    x1, y1, x2, y2 = new_room

    for ex1, ey1, ex2, ey2 in existing_rooms:
        # Check if they can be adjacent (allow shared walls)
        # If rooms can be adjacent, only need 1 cell spacing for wall
        wall_margin = 1
        if not (x2 + wall_margin < ex1 or x1 > ex2 + wall_margin or
                y2 + wall_margin < ey1 or y1 > ey2 + wall_margin):
            return True
    return False

def _force_place_small_room(n: int, existing_rooms: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Force place a small room"""
    min_size = 4
    max_size = min(6, n // 4)

    for size in range(min_size, max_size + 1):
        for x in range(1, n - size - 1):  # Ensure room doesn't reach boundary
            for y in range(1, n - size - 1):  # Ensure room doesn't reach boundary
                room = (x, y, x + size - 1, y + size - 1)
                if not _rooms_overlap_with_walls(room, existing_rooms):
                    return room

    # Last resort: place minimum room (4x4)
    if n >= 6:  # Ensure enough space for 4x4 room plus walls
        return (1, 1, 4, 4)
    return None

def _generate_connections(rooms: List[Tuple[int, int, int, int]], topology: str = "tree") -> List[Tuple[int, int]]:
    """Generate connections between rooms based on specified topology"""
    if len(rooms) <= 1:
        return []

    if topology == "line":
        return _generate_line_connections(rooms)
    elif topology == "star":
        return _generate_star_connections(rooms)
    else:  # Default to tree topology
        return _generate_tree_connections(rooms)

def _generate_line_connections(rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Generate linear connections between rooms (1→2→3→4)"""
    connections = []
    for i in range(len(rooms) - 1):
        connections.append((i, i + 1))
    return connections

def _generate_star_connections(rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Generate star-like connections between rooms (1→2, 1→3, 1→4)"""
    connections = []
    # Connect room 0 (first room) to all other rooms
    for i in range(1, len(rooms)):
        connections.append((0, i))
    return connections

def _generate_tree_connections(rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Generate tree structure connections between rooms (no cycles), using minimum spanning tree algorithm"""
    if len(rooms) <= 1:
        return []

    # Calculate room center points
    centers = []
    for x1, y1, x2, y2 in rooms:
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    # Use Prim's algorithm to generate minimum spanning tree, ensure no cycles
    visited = [False] * len(rooms)
    visited[0] = True  # Start from first room
    connections = []

    while len(connections) < len(rooms) - 1:
        min_dist = float('inf')
        best_edge = None

        # Find shortest connection from visited rooms to unvisited rooms
        for i in range(len(rooms)):
            if not visited[i]:
                continue
            for j in range(len(rooms)):
                if visited[j]:
                    continue

                # Calculate Manhattan distance
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                dist = abs(cx1 - cx2) + abs(cy1 - cy2)

                if dist < min_dist:
                    min_dist = dist
                    best_edge = (i, j)

        if best_edge:
            i, j = best_edge
            visited[j] = True
            connections.append(best_edge)
        else:
            break  # Cannot find more connections

    return connections

def _add_walls_around_rooms(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]]):
    """Add walls around all rooms, rooms don't reach boundary so there's always space for walls"""
    n = grid.shape[0]

    for i, room in enumerate(rooms):
        x1, y1, x2, y2 = room
        current_room_id = i + 1

        # Since rooms don't reach boundary, can always add walls around them
        # Top wall
        for x in range(max(0, x1-1), min(n, x2+2)):
            pos_value = grid[y1-1, x]
            # Set wall in impassable area or other room boundaries
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y1-1, x] = 0

        # Bottom wall
        for x in range(max(0, x1-1), min(n, x2+2)):
            pos_value = grid[y2+1, x]
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y2+1, x] = 0

        # Left wall
        for y in range(max(0, y1-1), min(n, y2+2)):
            pos_value = grid[y, x1-1]
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y, x1-1] = 0

        # Right wall
        for y in range(max(0, y1-1), min(n, y2+2)):
            pos_value = grid[y, x2+1]
            if pos_value == -1 or (1 <= pos_value <= len(rooms) and pos_value != current_room_id):
                grid[y, x2+1] = 0

    # Second pass: ensure all rooms are surrounded by walls or doors, cannot be directly adjacent to -1
    _ensure_rooms_surrounded_by_walls_or_doors(grid, rooms)

def _ensure_rooms_surrounded_by_walls_or_doors(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]]):
    """Ensure all rooms are surrounded by walls or doors, rooms cannot be directly adjacent to -1"""
    n = grid.shape[0]

    for i, room in enumerate(rooms):
        x1, y1, x2, y2 = room
        current_room_id = i + 1

        # Check each boundary position of the room
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if grid[y, x] == current_room_id:
                    # Check four directions of room cell
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx

                        # If adjacent position is within grid
                        if 0 <= ny < n and 0 <= nx < n:
                            neighbor_value = grid[ny, nx]

                            # If room is directly adjacent to -1, need to add wall in between
                            if neighbor_value == -1:
                                grid[ny, nx] = 0  # Set as wall

def _add_doors_between_rooms(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]],
                            connections: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Add doors between connected rooms, return list of successfully added doors"""
    doors_added = []

    for room1_idx, room2_idx in connections:
        room1 = rooms[room1_idx]
        room2 = rooms[room2_idx]

        # Find door position between two rooms
        door_pos = _find_door_between_rooms(grid, room1, room2)
        if door_pos:
            x, y, door_type = door_pos
            grid[y, x] = door_type
            doors_added.append(door_pos)

    return doors_added

def _find_door_between_rooms(grid: np.ndarray, room1: Tuple[int, int, int, int],
                            room2: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Find door position between two rooms"""
    x1_1, y1_1, x2_1, y2_1 = room1
    x1_2, y1_2, x2_2, y2_2 = room2

    # Check horizontal adjacency (east-west door)
    if abs(x2_1 + 1 - x1_2) <= 1:  # room1 on left, room2 on right
        door_x = x2_1 + 1 if x2_1 + 1 == x1_2 else (x2_1 + x1_2) // 2
        y_overlap_start = max(y1_1, y1_2)
        y_overlap_end = min(y2_1, y2_2)

        if y_overlap_end >= y_overlap_start:
            # Avoid placing door in corner
            if y_overlap_end > y_overlap_start:
                door_y = random.randint(y_overlap_start, y_overlap_end)
            else:
                door_y = y_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 101)  # East-west door

    elif abs(x2_2 + 1 - x1_1) <= 1:  # room2 on left, room1 on right
        door_x = x2_2 + 1 if x2_2 + 1 == x1_1 else (x2_2 + x1_1) // 2
        y_overlap_start = max(y1_1, y1_2)
        y_overlap_end = min(y2_1, y2_2)

        if y_overlap_end >= y_overlap_start:
            if y_overlap_end > y_overlap_start:
                door_y = random.randint(y_overlap_start, y_overlap_end)
            else:
                door_y = y_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 101)  # East-west door

    # Check vertical adjacency (north-south door)
    if abs(y2_1 + 1 - y1_2) <= 1:  # room1 on top, room2 on bottom
        door_y = y2_1 + 1 if y2_1 + 1 == y1_2 else (y2_1 + y1_2) // 2
        x_overlap_start = max(x1_1, x1_2)
        x_overlap_end = min(x2_1, x2_2)

        if x_overlap_end >= x_overlap_start:
            if x_overlap_end > x_overlap_start:
                door_x = random.randint(x_overlap_start, x_overlap_end)
            else:
                door_x = x_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 100)  # North-south door

    elif abs(y2_2 + 1 - y1_1) <= 1:  # room2 on top, room1 on bottom
        door_y = y2_2 + 1 if y2_2 + 1 == y1_1 else (y2_2 + y1_1) // 2
        x_overlap_start = max(x1_1, x1_2)
        x_overlap_end = min(x2_1, x2_2)

        if x_overlap_end >= x_overlap_start:
            if x_overlap_end > x_overlap_start:
                door_x = random.randint(x_overlap_start, x_overlap_end)
            else:
                door_x = x_overlap_start

            if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                return (door_x, door_y, 100)  # North-south door

    return None

def _verify_connectivity(grid: np.ndarray, num_rooms: int) -> bool:
    """Verify if all rooms are connected"""
    if num_rooms <= 1:
        return True

    # Find positions of all rooms
    room_positions = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if 1 <= grid[i, j] <= num_rooms:
                room_id = grid[i, j]
                if room_id not in room_positions:
                    room_positions[room_id] = []
                room_positions[room_id].append((i, j))

    # Check if all rooms exist
    if len(room_positions) != num_rooms:
        return False

    # Use BFS to check if all other rooms can be reached from room 1
    visited_rooms = set()
    start_pos = room_positions[1][0]  # Start from first position of room 1
    queue = [start_pos]
    visited_positions = set([start_pos])

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, left, right

    while queue:
        y, x = queue.pop(0)
        current_value = grid[y, x]

        # If current position is a room, record visited room
        if 1 <= current_value <= num_rooms:
            visited_rooms.add(current_value)

        # Explore adjacent positions
        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            # Check boundaries
            if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and
                (ny, nx) not in visited_positions):

                next_value = grid[ny, nx]

                # Can pass through rooms or doors
                if (1 <= next_value <= num_rooms) or next_value == 100 or next_value == 101:
                    visited_positions.add((ny, nx))
                    queue.append((ny, nx))

    # Check if all rooms were visited
    return len(visited_rooms) == num_rooms

def _verify_no_rooms_at_boundary(grid: np.ndarray, num_rooms: int) -> bool:
    """Verify no rooms reach grid boundary"""
    n = grid.shape[0]

    # Check four boundaries
    for i in range(n):
        # Top and bottom boundaries
        if 1 <= grid[0, i] <= num_rooms or 1 <= grid[n-1, i] <= num_rooms:
            return False
        # Left and right boundaries
        if 1 <= grid[i, 0] <= num_rooms or 1 <= grid[i, n-1] <= num_rooms:
            return False

    return True

def _grid_to_emoji(grid: np.ndarray) -> str:
    """Convert grid to emoji display"""
    emoji_map = {
        -1: "⬛",  # Impassable area - black square
        0: "🧱",   # Wall - brick
        100: "🚪", # North-south door - door
        101: "🚪", # East-west door - door
    }

    # Rooms use different colored squares
    room_emojis = ["🟦", "🟩", "🟨", "🟪", "🟧", "🟫", "⬜", "🟥"]

    result = []
    for row in grid:
        emoji_row = []
        for val in row:
            if val == -1:
                emoji_row.append("⬛")
            elif val == 0:
                emoji_row.append("🧱")
            elif val == 100 or val == 101:
                emoji_row.append("🚪")
            elif 1 <= val <= len(room_emojis):
                emoji_row.append(room_emojis[val - 1])
            else:
                emoji_row.append("❓")  # Unknown value
        result.append("".join(emoji_row))

    return "\n".join(result)

# Test function
if __name__ == "__main__":
    # Test different topologies
    topologies = ["tree", "line", "star"]

    for topology in topologies:
        print(f"\n{'='*50}")
        print(f"🏗️  Testing {topology.upper()} topology")
        print(f"{'='*50}")

        for level in range(1, 5):  # Test levels 1-3 for multiple rooms
            print(f"\n🔸 Level {level} ({level+1} rooms) - {topology} topology")

            # Use consistent seed for comparison
            seed = 43
            main_size = 8 if level > 0 else None

            grid = generate_rooms_auto(level, main=main_size, seed=seed, topology=topology)

            if main_size:
                print(f"🎯 Specified main room size: {main_size}×{main_size}")

            unique_values = sorted(set(grid.flatten()))
            print(f"📊 Unique values in grid: {unique_values}")

            # Count rooms and doors
            room_count = sum(1 for v in unique_values if v > 0 and v < 100)
            door_count = sum(1 for v in unique_values if v >= 100)
            print(f"🏠 Actual room count: {room_count}, 🚪 Door count: {door_count}")

            # Print emoji version
            print("🎨 Emoji version:")
            print(_grid_to_emoji(grid))
