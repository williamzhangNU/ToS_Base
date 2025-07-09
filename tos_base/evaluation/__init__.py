"""
Evaluation tasks and utilities for spatial reasoning.
"""

from .tasks import BaseEvaluationTask
from .task_factory import get_eval_task

__all__ = ['BaseEvaluationTask', 'get_eval_task'] 