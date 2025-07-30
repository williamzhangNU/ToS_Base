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


# 测试函数
if __name__ == "__main__":
    # 测试用例1: 标准JSON格式
    test_response1 = '''
    根据描述，房间布局如下：
    ```json
    {
      "whiteboard": {"position": [1, 1], "facing": "east"},
      "oven": {"position": [0, 1], "facing": "east"},
      "chair": {"position": [-1, 1], "facing": "east"}
    }
    ```
    '''

    print("=== 测试用例1: 标准JSON格式 ===")
    room1 = extract_json_and_create_room(test_response1, "test_room1")
    if room1:
        print("成功创建Room对象:")
        print(room1)
        print("\n原始可视化:")
        room1.plot('text')
        print("\n改进的可视化:")
        plot_room_with_orientations(room1)
    else:
        print("创建Room对象失败")

    # 测试用例2: 不同朝向
    test_response2 = '''
    房间中的物品位置：
    ```json
    {
      "table": {"position": [2, 0], "facing": "north"},
      "sofa": {"position": [-1, -1], "facing": "west"},
      "lamp": {"position": [0, 2], "facing": "south"}
    }
    ```
    '''

    print("\n=== 测试用例2: 不同朝向 ===")
    room2 = extract_json_and_create_room(test_response2, "test_room2")
    if room2:
        print("成功创建Room对象:")
        print(room2)
        print("\n改进的可视化:")
        plot_room_with_orientations(room2)
    else:
        print("创建Room对象失败")

    # 测试用例3: 没有朝向信息
    test_response3 = '''
    ```json
    {
      "desk": {"position": [1, 0]},
      "bookshelf": {"position": [-2, 1]}
    }
    ```
    '''

    print("\n=== 测试用例3: 没有朝向信息 ===")
    room3 = extract_json_and_create_room(test_response3, "test_room3")
    if room3:
        print("成功创建Room对象:")
        print(room3)
        print("\n改进的可视化:")
        plot_room_with_orientations(room3)
    else:
        print("创建Room对象失败")
