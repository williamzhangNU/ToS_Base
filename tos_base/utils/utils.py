"""
JSON提取器模块

该模块提供从模型回答中提取JSON格式的房间布局信息，并创建Room对象的功能。

主要功能：
1. extract_json_and_create_room: 从模型回答中提取JSON并创建Room对象
2. extract_json_from_text: 从文本中提取JSON内容
3. create_room_from_json: 从JSON数据创建Room对象
4. validate_room_json: 验证JSON数据格式

支持的JSON格式示例：
```json
{
  "whiteboard": {"position": [1, 1], "facing": "east"},
  "oven": {"position": [0, 1], "facing": "east"},
  "chair": {"position": [-1, 1], "facing": "east"}
}
```

支持的朝向：north, south, east, west
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# 添加路径以支持相对导入
try:
    from ..core.room import Room
    from ..core.object import Object, Agent
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.room import Room
    from core.object import Object, Agent


def extract_json_and_create_room(model_response: str, room_name: str = "extracted_room", use_improved_plot: bool = True) -> Optional[Room]:
    """
    从模型回答中提取JSON格式的房间布局信息，并创建Room对象

    Args:
        model_response: 模型的回答文本，包含JSON格式的房间布局
        room_name: 房间名称，默认为"extracted_room"
        use_improved_plot: 是否使用改进的可视化函数，默认为True

    Returns:
        Room对象，如果提取失败则返回None

    Example:
        >>> response = '''
        ... 根据描述，房间布局如下：
        ... ```json
        ... {
        ...   "whiteboard": {"position": [1, 1], "facing": "east"},
        ...   "oven": {"position": [0, 1], "facing": "east"},
        ...   "chair": {"position": [-1, 1], "facing": "east"}
        ... }
        ... ```
        ... '''
        >>> room = extract_json_and_create_room(response)
        >>> room.plot_improved()  # 使用改进的可视化
    """

    # 提取JSON内容
    json_data = extract_json_from_text(model_response)
    if json_data is None:
        return None

    # 创建Room对象
    room = create_room_from_json(json_data, room_name)

    # 为Room对象添加改进的可视化方法
    if room and use_improved_plot:
        room.plot_improved = lambda: plot_room_with_orientations(room)

    return room


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取JSON内容
    
    Args:
        text: 包含JSON的文本
    
    Returns:
        解析后的JSON字典，如果提取失败则返回None
    """
    
    # 尝试多种JSON提取模式
    patterns = [
        # 标准的```json代码块
        r'```json\s*\n(.*?)\n\s*```',
        # 没有语言标识的代码块
        r'```\s*\n(.*?)\n\s*```',
        # 直接的JSON对象（以{开始，以}结束）
        r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # 清理匹配的文本
                json_text = match.strip()
                # 尝试解析JSON
                json_data = json.loads(json_text)
                if isinstance(json_data, dict) and json_data:
                    return json_data
            except (json.JSONDecodeError, ValueError):
                continue
    
    return None


def create_room_from_json(json_data: Dict[str, Any], room_name: str = "extracted_room") -> Optional[Room]:
    """
    从JSON数据创建Room对象
    
    Args:
        json_data: 包含房间布局信息的JSON字典
        room_name: 房间名称
    
    Returns:
        Room对象，如果创建失败则返回None
    """
    
    try:
        objects = []
        
        # 方向映射：字符串到numpy数组
        direction_mapping = {
            "north": np.array([0, 1]),
            "south": np.array([0, -1]),
            "east": np.array([1, 0]),
            "west": np.array([-1, 0])
        }
        
        # 遍历JSON中的每个对象
        for obj_name, obj_info in json_data.items():
            # 跳过agent相关的对象
            if obj_name.lower() in ['agent', 'you', 'player']:
                continue
                
            # 获取位置信息
            if 'position' not in obj_info:
                print(f"警告: 对象 {obj_name} 缺少位置信息")
                continue
                
            position = obj_info['position']
            if not isinstance(position, list) or len(position) != 2:
                print(f"警告: 对象 {obj_name} 的位置格式不正确: {position}")
                continue
                
            pos = np.array([float(position[0]), float(position[1])])
            
            # 获取朝向信息
            facing = obj_info.get('facing', 'north')
            if isinstance(facing, str):
                facing = facing.lower()
                if facing in direction_mapping:
                    ori = direction_mapping[facing]
                else:
                    print(f"警告: 对象 {obj_name} 的朝向 '{facing}' 不被识别，使用默认朝向 'north'")
                    ori = direction_mapping['north']
            else:
                print(f"警告: 对象 {obj_name} 的朝向格式不正确: {facing}，使用默认朝向 'north'")
                ori = direction_mapping['north']
            
            # 创建对象
            obj = Object(name=obj_name, pos=pos, ori=ori, has_orientation=True)
            objects.append(obj)
        
        if not objects:
            print("警告: 没有找到有效的对象")
            return None
        
        # 创建默认的agent
        agent = Agent(name="agent", pos=np.array([0, 0]), ori=np.array([0, 1]))
        
        # 创建Room对象
        room = Room(agent=agent, objects=objects, name=room_name)

        return room
        
    except Exception as e:
        print(f"创建Room对象时发生错误: {e}")
        return None


def validate_room_json(json_data: Dict[str, Any]) -> bool:
    """
    验证JSON数据是否符合房间布局的格式要求
    
    Args:
        json_data: 要验证的JSON字典
    
    Returns:
        True如果格式正确，False否则
    """
    
    if not isinstance(json_data, dict):
        return False
    
    valid_directions = {'north', 'south', 'east', 'west'}
    
    for obj_name, obj_info in json_data.items():
        if not isinstance(obj_info, dict):
            return False
            
        # 检查位置信息
        if 'position' not in obj_info:
            return False
            
        position = obj_info['position']
        if not isinstance(position, list) or len(position) != 2:
            return False
            
        try:
            float(position[0])
            float(position[1])
        except (ValueError, TypeError):
            return False
        
        # 检查朝向信息（可选）
        if 'facing' in obj_info:
            facing = obj_info['facing']
            if not isinstance(facing, str) or facing.lower() not in valid_directions:
                return False
    
    return True


def plot_room_with_orientations(room):
    """
    改进的房间可视化函数，更好地显示对象朝向
    """
    min_x, max_x, min_y, max_y = room.get_boundary()
    min_x, max_x, min_y, max_y = int(min_x)-1, int(max_x)+1, int(min_y)-1, int(max_y)+1

    width, height = max_x - min_x + 1, max_y - min_y + 1
    grid = [[' '] * width for _ in range(height)]

    ori_map = {(0,1):'^', (0,-1):'v', (1,0):'>', (-1,0):'<'}
    labels = []

    # 首先放置所有对象
    for i, obj in enumerate([room.agent] + room.objects):
        x, y = int(obj.pos[0]) - min_x, max_y - int(obj.pos[1])
        symbol = 'A' if obj == room.agent else str(i-1)
        labels.append(f"{symbol}:{obj.name}")

        if 0 <= y < height and 0 <= x < width:
            # 将对象符号和朝向符号组合显示
            ori_symbol = ori_map.get(tuple(obj.ori), '?')
            grid[y][x] = f"{symbol}{ori_symbol}"

    print(f"\n--- {room.name} (改进版) ---")
    print("Legend:", " | ".join(labels))
    print("符号格式: [对象][朝向] (^:北 v:南 >:东 <:西)")
    for y in range(height):
        print(f"{max_y-y:3d} " + "".join(f"{cell:>3}" for cell in grid[y]))
    print("    " + "".join(f"{min_x+x:>3}" for x in range(width)))


def create_room_and_visualize(model_response: str, room_name: str = "extracted_room") -> Optional[Room]:
    """
    便捷函数：提取JSON，创建Room对象，并立即显示改进的可视化

    Args:
        model_response: 模型的回答文本，包含JSON格式的房间布局
        room_name: 房间名称，默认为"extracted_room"

    Returns:
        Room对象，如果提取失败则返回None
    """
    room = extract_json_and_create_room(model_response, room_name)
    if room:
        print(f"成功创建Room对象: {room_name}")
        plot_room_with_orientations(room)
        return room
    else:
        print("创建Room对象失败")
        return None


def compare_room_consistency(room1: Room, room2: Room):
    """
    比较两个房间中对象的相对关系和朝向一致性

    Args:
        room1: 第一个房间（通常是预测的房间）
        room2: 第二个房间（通常是真实的grounded房间）

    Returns:
        tuple: (相对关系一致性比例, 朝向一致性比例)
               - 相对关系一致性比例: 0.0-1.0，表示两两物体相对关系一致的比例
               - 朝向一致性比例: 0.0-1.0，表示各物体朝向一致的比例
    """

    # 获取两个房间中共同的对象名称（包括agent，但排除initial_pos）
    room1_obj_names = {obj.name for obj in room1.objects}
    room2_obj_names = {obj.name for obj in room2.objects}
    common_obj_names = room1_obj_names.intersection(room2_obj_names)

    # 添加agent到共同对象列表中（agent总是存在于两个房间中）
    common_obj_names.add('agent')

    if len(common_obj_names) < 2:
        print("警告: 共同对象少于2个，无法进行相对关系比较")
        return 0.0, 0.0

    # 计算相对关系一致性
    total_pairs = 0
    consistent_pairs = 0

    # 遍历所有对象对
    common_obj_list = list(common_obj_names)
    for i in range(len(common_obj_list)):
        for j in range(i + 1, len(common_obj_list)):
            obj1_name = common_obj_list[i]
            obj2_name = common_obj_list[j]

            try:
                # 获取两个房间中的相对关系
                dir_pair1, _ = room1.get_direction(obj1_name, obj2_name, perspective='allo')
                dir_pair2, _ = room2.get_direction(obj1_name, obj2_name, perspective='allo')

                # 比较相对关系是否一致
                if dir_pair1.horiz == dir_pair2.horiz and dir_pair1.vert == dir_pair2.vert:
                    consistent_pairs += 1

                total_pairs += 1

            except Exception as e:
                print(f"警告: 比较对象 {obj1_name} 和 {obj2_name} 的相对关系时出错: {e}")
                continue

    # 计算相对关系一致性比例
    relation_consistency = consistent_pairs / total_pairs if total_pairs > 0 else 0.0

    # 计算朝向一致性（排除agent）
    total_orientations = 0
    consistent_orientations = 0

    for obj_name in common_obj_names:
        # 跳过agent的朝向比较
        if obj_name == 'agent':
            continue

        try:
            obj1 = room1.get_object_by_name(obj_name)
            obj2 = room2.get_object_by_name(obj_name)

            # 只比较有朝向的对象
            if obj1.has_orientation and obj2.has_orientation:
                # 比较朝向向量是否相同
                if np.array_equal(obj1.ori, obj2.ori):
                    consistent_orientations += 1

                total_orientations += 1

        except Exception as e:
            print(f"警告: 比较对象 {obj_name} 的朝向时出错: {e}")
            continue

    # 计算朝向一致性比例
    orientation_consistency = consistent_orientations / total_orientations if total_orientations > 0 else 0.0
    result = {}
    result["directional_similarity"] = relation_consistency
    result["facing_similarity"] = orientation_consistency
    result["overall_similarity"] =  0.7 * relation_consistency + 0.3 * orientation_consistency
    return result


# 测试函数
if __name__ == "__main__":
    print("\n=== 测试用例4: 房间一致性比较 ===")

    # 创建两个相似的房间进行比较
    # 房间A: 原始房间
    test_response_a = '''
    ```json
    {
      "table": {"position": [2, 1], "facing": "north"},
      "chair": {"position": [0, 1], "facing": "east"},
      "lamp": {"position": [-1, 2], "facing": "south"}
    }
    ```
    '''

    # 房间B: 完全相同的房间
    test_response_b = '''
    ```json
    {
      "table": {"position": [2, 1], "facing": "north"},
      "chair": {"position": [0, 1], "facing": "east"},
      "lamp": {"position": [-1, 2], "facing": "south"}
    }
    ```
    '''

    # 房间C: 部分不同的房间（位置改变）
    test_response_c = '''
    ```json
    {
      "table": {"position": [1, 2], "facing": "north"},
      "chair": {"position": [0, 1], "facing": "east"},
      "lamp": {"position": [-1, 2], "facing": "south"}
    }
    ```
    '''

    # 房间D: 朝向不同的房间
    test_response_d = '''
    ```json
    {
      "table": {"position": [4, 1], "facing": "south"},
      "chair": {"position": [0, 1], "facing": "west"},
      "lamp": {"position": [-1, 2], "facing": "north"}
    }
    ```
    '''

    room_a = extract_json_and_create_room(test_response_a, "room_a")
    room_b = extract_json_and_create_room(test_response_b, "room_b")
    room_c = extract_json_and_create_room(test_response_c, "room_c")
    room_d = extract_json_and_create_room(test_response_d, "room_d")
    plot_room_with_orientations(room_a)
    plot_room_with_orientations(room_b)
    plot_room_with_orientations(room_c)
    plot_room_with_orientations(room_d)
    if all([room_a, room_b, room_c, room_d]):
        print("成功创建所有测试房间")

        # 测试完全相同的房间
        result = compare_room_consistency(room_a, room_b)
        print(f"\n房间A vs 房间B (完全相同):")
        print(f"相对关系一致性: {result['directional_similarity']:.2f}")
        print(f"朝向一致性: {result['facing_similarity']:.2f}")
        print(f"总体相似度: {result['overall_similarity']:.2f}")

        # 测试位置不同的房间
        result = compare_room_consistency(room_a, room_c)
        print(f"\n房间A vs 房间C (位置部分不同):")
        print(f"相对关系一致性: {result['directional_similarity']:.2f}")
        print(f"朝向一致性: {result['facing_similarity']:.2f}")
        print(f"总体相似度: {result['overall_similarity']:.2f}")

        # 测试朝向不同的房间
        result = compare_room_consistency(room_a, room_d)
        print(f"\n房间A vs 房间D (朝向不同):")
        print(f"相对关系一致性: {result['directional_similarity']:.2f}")
        print(f"朝向一致性: {result['facing_similarity']:.2f}")
        print(f"总体相似度: {result['overall_similarity']:.2f}")

    else:
        print("创建测试房间失败")
