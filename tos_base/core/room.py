import numpy as np
import json
from typing import List, Union, Dict, Any, Tuple
import copy

from .object import Object, Agent
from .relationship import DirPair, DirectionSystem, Dir
from .graph import DirectionalGraph
    


class Room:
    """
    Simplified Room class focused on state representation.
    Handles basic object management and spatial relationships.
    """

    def __init__(self, objects: List[Object], name: str = 'room', agent: Agent = None):
        self.name = name
        self._init_objects(objects, agent)

    def _init_objects(self, objects: List[Object], agent: Agent = None):
        """Initialize objects, agent, and validate unique names"""
        self.objects = copy.deepcopy(objects)
        self.agent = copy.deepcopy(agent) if agent is not None else None
        self.all_objects = ([self.agent] + self.objects) if self.agent else self.objects
        self.gt_graph = DirectionalGraph(self.all_objects, is_explore=False)
        
        # Validate unique names
        names = [obj.name for obj in self.all_objects]
        assert len(names) == len(set(names)), "All object names must be unique"

    def add_object(self, obj: Object):
        """Add an object to the room"""
        self._init_objects(self.objects + [obj], self.agent)
    
    def remove_object(self, obj_name: str):
        """Remove an object from the room"""
        self._init_objects([o for o in self.objects if o.name != obj_name], self.agent)

    def get_object_by_name(self, name: str) -> Object:
        """Get object by name"""
        for obj in self.all_objects:
            if obj.name == name:
                return obj
        raise ValueError(f"Object '{name}' not found in room")

    def has_object(self, name: str) -> bool:
        """Check if object exists in room"""
        return any(obj.name == name for obj in self.all_objects)

    def get_direction(self, obj1_name: str, obj2_name: str, 
                     anchor_name: str = None, perspective: str = None) -> Tuple[DirPair, str]:
        """Get spatial relationship between two objects"""
        obj1 = self.get_object_by_name(obj1_name)
        obj2 = self.get_object_by_name(obj2_name)
        perspective = perspective or ('ego' if self.agent else 'allo')
        
        anchor_ori = None
        if anchor_name:
            anchor = self.get_object_by_name(anchor_name)
            assert anchor.has_orientation, "Anchor must have orientation"
            anchor_ori = anchor.ori
        
        dir_pair = DirectionSystem.get_direction(obj1.pos, obj2.pos, anchor_ori)
        dir_str = DirectionSystem.to_string(dir_pair, perspective=perspective)
        
        return dir_pair, dir_str

    def get_orientation(self, obj_name: str, anchor_name: str) -> Tuple[DirPair, str]:
        """Get orientation of an object relative to an anchor."""
        obj = self.get_object_by_name(obj_name)
        anchor = self.get_object_by_name(anchor_name)
        assert anchor.has_orientation, "Anchor must have orientation"

        dir_pair = DirectionSystem.get_relative_orientation(tuple(obj.ori), tuple(anchor.ori))

        mapping = {DirPair(Dir.SAME, Dir.FORWARD): 'away from you',
                   DirPair(Dir.SAME, Dir.BACKWARD): 'towards you',
                   DirPair(Dir.RIGHT, Dir.SAME): 'to your right',
                   DirPair(Dir.LEFT, Dir.SAME): 'to your left'
        }

        ori_str = mapping[dir_pair]
            
        return dir_pair, ori_str

    def get_room_description(self) -> str:
        """Get textual description of the room"""
        if self.agent:
            desc = f"Imagine yourself as {self.agent.name} in a room.\n"
            desc += "You are facing north.\n"
            desc += f"Objects in the room: {', '.join([obj.name for obj in self.objects])}\n"
        else:
            desc = "Imagine looking at a room from above.\n"
            desc += f"Objects in the room: {', '.join([obj.name for obj in self.objects])}\n"
        return desc
    
    def get_boundary(self):
        """
        Get the boundary of the room
        """
        positions = np.array([obj.pos for obj in self.all_objects])
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        # Generate random position within extended boundaries
        # x ranges from (2*min_x - max_x) to (2*max_x - min_x)
        # y ranges from (2*min_y - max_y) to (2*max_y - min_y)
        min_x_bound = min_x - min(max_x - min_x, 1)
        max_x_bound = max_x + min(max_x - min_x, 1)
        min_y_bound = min_y - min(max_y - min_y, 1)
        max_y_bound = max_y + min(max_y - min_y, 1)
        return min_x_bound, max_x_bound, min_y_bound, max_y_bound
    
    def get_objects_orientation(self):
        """
        Get the objects in the room with their orientation
        """
        ori_mapping = {
            (0, 1): "north",
            (0, -1): "south",
            (1, 0): "east",
            (-1, 0): "west",
        }
        desc = "Orientation of objects in the room are: \n"
        for obj in self.objects:
            desc += f"{obj.name} facing {ori_mapping[tuple(obj.ori)]}\n"
        return desc

    def copy(self) -> 'Room':
        """Create a deep copy of the room"""
        return Room(
            objects=copy.deepcopy(self.objects),
            name=self.name,
            agent=copy.deepcopy(self.agent)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize room to dictionary"""
        return {
            'name': self.name,
            'objects': [obj.to_dict() for obj in self.objects],
            'agent': self.agent.to_dict() if self.agent else None,
            'all_objects': [obj.to_dict() for obj in self.all_objects]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Room':
        """Deserialize room from dictionary"""
        objects = [Object.from_dict(obj_data) for obj_data in data['objects']]
        agent = Agent.from_dict(data['agent']) if data['agent'] else None
        return cls(objects=objects, name=data['name'], agent=agent)
    
    def plot(self, render_mode: str = 'text', save_path: str = None):
        """Plot the room with objects and orientations"""
        if render_mode == 'text':
            self._plot_text()
        elif render_mode == 'img':
            self._plot_img(save_path)
        else:
            raise ValueError(f"render_mode must be 'text' or 'img', got '{render_mode}'")
    
    def _plot_text(self):
        """Text grid visualization"""
        min_x, max_x, min_y, max_y = self.get_boundary()
        min_x, max_x, min_y, max_y = int(min_x)-1, int(max_x)+1, int(min_y)-1, int(max_y)+1
        
        width, height = max_x - min_x + 1, max_y - min_y + 1
        grid = [[' '] * width for _ in range(height)]
        
        ori_map = {(0,1):'^', (0,-1):'v', (1,0):'>', (-1,0):'<'}
        labels = []
        
        for i, obj in enumerate(self.all_objects):
            x, y = int(obj.pos[0]) - min_x, max_y - int(obj.pos[1])
            symbol = 'A' if obj == self.agent else str(i if self.agent is None else i-1)
            labels.append(f"{symbol}:{obj.name}")
            
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = symbol
                # Place orientation arrow
                dx, dy = obj.ori
                ax, ay = x + dx, y - dy
                if 0 <= ay < height and 0 <= ax < width and grid[ay][ax] == ' ':
                    grid[ay][ax] = ori_map.get(tuple(obj.ori), '?')
        
        print(f"\n--- {self.name} ---")
        print("Legend:", " | ".join(labels))
        for y in range(height):
            print(f"{max_y-y:3d} " + "".join(f"{cell} " for cell in grid[y]))
        print("    " + "".join(f"{min_x+x} " for x in range(width)))
    
    def _plot_img(self, save_path=None):
        """Matplotlib visualization"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for img mode")
            return
        
        if save_path is None:
            save_path = f"room_{self.name}.pdf"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        min_x, max_x, min_y, max_y = self.get_boundary()
        
        for i, obj in enumerate(self.all_objects):
            x, y = obj.pos
            color = plt.cm.tab10(i)
            marker = 's' if obj == self.agent else 'o'
            
            ax.scatter(x, y, c=[color], s=120, marker=marker, edgecolor='black')
            ax.annotate(obj.name, (x, y), xytext=(3, 3), textcoords='offset points')
            ax.arrow(x, y, obj.ori[0]*0.7, obj.ori[1]*0.7, 
                    head_width=0.15, fc=color, ec='black')
        
        ax.set_xlim(min_x-1, max_x+1)
        ax.set_ylim(min_y-1, max_y+1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f'Room: {self.name}')
        
        # Save to specified file path
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Room plot saved to: {save_path}")
        plt.close()

    def __repr__(self):
        objects_details = [f"{obj.name}@{tuple(obj.pos)}:{tuple(obj.ori)}" for obj in self.objects]
        agent_detail = f"{self.agent.name}@{tuple(self.agent.pos)}:{tuple(self.agent.ori)}" if self.agent else None
        return f"Room(name={self.name}, objects=[{', '.join(objects_details)}], agent={agent_detail})"



if __name__ == '__main__':
    # Test plot function
    print("Testing Room plot function...")
    
    # Create test objects
    objects = [
        Object(name="table", pos=np.array([2, 1]), ori=np.array([1, 0])),   # facing east
        Object(name="chair", pos=np.array([0, 3]), ori=np.array([0, -1])),  # facing south
        Object(name="sofa", pos=np.array([-1, 0]), ori=np.array([-1, 0])), # facing west
        Object(name="lamp", pos=np.array([1, -2]), ori=np.array([0, 1])),  # facing north
    ]
    
    # Test with agent
    agent = Agent(name="agent")
    agent.pos = np.array([0, 0])
    agent.ori = np.array([0, 1])  # facing north
    
    room_with_agent = Room(objects=objects, name="test_room_with_agent", agent=agent)
    
    # print("\n=== Room with Agent - Text ===")
    # room_with_agent.plot('text')
    
    # print("\n=== Room with Agent - Image ===")
    # room_with_agent.plot('img', 'test_with_agent.pdf')
    
    # # Test without agent
    # room_no_agent = Room(objects=objects, name="test_room_no_agent", agent=None)
    
    # print("\n=== Room without Agent - Text ===")
    # room_no_agent.plot('text')
    
    # print("\n=== Room without Agent - Image ===")
    # room_no_agent.plot('img', 'test_no_agent.pdf')
    
    # # Test error handling
    # print("\n=== Error Handling Test ===")
    # try:
    #     room_with_agent.plot('invalid_mode')
    # except ValueError as e:
    #     print(f"✓ Caught expected error: {e}")

    # test add and remove object
    print("\n=== Add and Remove Object Test ===")
    room_with_agent.plot('text')
    room_with_agent.add_object(Object(name="new_obj", pos=np.array([5, 5]), ori=np.array([0, 1])))
    print(room_with_agent.objects, room_with_agent.all_objects, room_with_agent.gt_graph._v_matrix)
    room_with_agent.plot('text')
    room_with_agent.remove_object(objects[0].name)
    print(room_with_agent.objects, room_with_agent.all_objects, room_with_agent.gt_graph._v_matrix)
    room_with_agent.plot('text')
    
    print("\nAll tests completed!")



