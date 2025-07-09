"""
Base module for spatial reasoning environment.

This module provides the core components for spatial reasoning tasks including:
- Core data structures (objects, rooms, relationships)
- Action system for agent interactions
- Exploration and evaluation management
- Utility functions for room generation and evaluation
"""

# Core data structures
from .core.object import Object, Agent
from .core.room import Room
from .core.relationship import DirPair, DirectionSystem, Dir
from .core.graph import DirectionalGraph
from .core.constant import (
    AGENT_NAME, 
    CANDIDATE_OBJECTS, 
    ADDITIONAL_CANDIDATE_OBJECTS,
    easy_room_config,
    easy_room_config_2,
    easy_room_config_3
)

# Action system
from .actions.base import BaseAction, ActionResult
from .actions.actions import (
    MoveAction,
    RotateAction, 
    ReturnAction,
    ObserveAction,
    TermAction,
    QueryAction,
    ActionSequence
)

# Managers
from .managers.exploration_manager import ExplorationManager
from .managers.evaluation_manager import EvaluationManager

# Evaluation tasks
from .evaluation.tasks import BaseEvaluationTask
from .evaluation.task_factory import get_eval_task

# Utilities
from .utils.room_utils import generate_room
from .utils.eval_utilities import *

__all__ = [
    # Core
    'Object', 'Agent', 'Room', 'DirPair', 'DirectionSystem', 'Dir', 'DirectionalGraph',
    'AGENT_NAME', 'CANDIDATE_OBJECTS', 'ADDITIONAL_CANDIDATE_OBJECTS',
    'easy_room_config', 'easy_room_config_2', 'easy_room_config_3',
    
    # Actions
    'BaseAction', 'ActionResult', 'ActionSequence',
    'MoveAction', 'RotateAction', 'ReturnAction', 'ObserveAction', 'TermAction', 'QueryAction',
    
    # Managers
    'ExplorationManager', 'EvaluationManager',
    
    # Evaluation
    'BaseEvaluationTask', 'get_eval_task',
    
    # Utils
    'generate_room'
]
