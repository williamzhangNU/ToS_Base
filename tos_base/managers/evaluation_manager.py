"""Simple Evaluation Manager for SpatialGym Environment"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..evaluation.task_types import EvalTaskType
from ..core.room import Room
from ..core.object import Agent
from ..evaluation.tasks import BaseEvaluationTask, EvaluationData

@dataclass
class EvaluationTurnLog:
    """Log data for a single evaluation turn."""
    task_type: str
    user_answer: str
    is_correct: bool
    evaluation_info: Dict[str, Any]
    evaluation_data: EvaluationData
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None

    def to_dict(self):
        evaluation_data = self.evaluation_data.to_dict()
        if "question" in evaluation_data:
            evaluation_data.pop("question")
        evaluation_data['choices'] = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(evaluation_data['choices'])])
        return {
            "task_type": self.task_type,
            "user_answer": self.user_answer,
            "is_correct": self.is_correct,
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "evaluation_info": self.evaluation_info,
            "evaluation_data": evaluation_data
        }


class EvaluationManager:
    """
    Manages evaluation tasks for the SpatialGym environment.
    
    Handles task initialization, question generation, answer evaluation,
    and tracking of evaluation results across multiple tasks.
    """
    DEFAULT_EVAL_SUMMARY = {
        "accuracy": 0.0,
        "total_tasks": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "unanswered_count": 0
    }
    
    def __init__(self, eval_tasks: List[Dict[str, Any]], np_random: np.random.Generator, room: Room, agent: Agent):
        self.eval_tasks = eval_tasks
        self.np_random = np_random
        self.room = room.copy()
        self.agent = agent.copy()
        self.results = []
        self.turn_logs: List[EvaluationTurnLog] = []
        
        # Initialize tasks
        self.tasks = []
        for task_spec in eval_tasks:
            task_type = task_spec['task_type']
            task_kwargs = task_spec.get('task_kwargs', {})
            task = EvalTaskType.create_task(task_type, np_random, room, agent, task_kwargs)
            self.tasks.append(task)
            self.results.append({
                "task_type": task.__class__.__name__,
                "correct": False,
                "info": {}
            })
        self.current_index = 0
    
    def _get_current_eval_task(self) -> Optional[BaseEvaluationTask]:
        """Get current evaluation task."""
        assert self.current_index < len(self.tasks), "No more tasks"
        return self.tasks[self.current_index]
    
    def get_current_question(self) -> Optional[str]:
        """Get question for current task."""
        task = self._get_current_eval_task()
        return None if task is None else task.question if task.question else task.generate_question()
    
    def evaluate_answer(self, answer: str) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate answer for current task."""
        assert self.current_index < len(self.tasks), "No more tasks"
        
        task = self.tasks[self.current_index]
        correct, info = task.evaluate(answer)
        
        # Record result
        self.results[self.current_index]["correct"] = correct
        self.results[self.current_index]["info"] = info
        
        # Create turn log
        turn_log = EvaluationTurnLog(
            task_type=task.__class__.__name__,
            user_answer=answer,
            is_correct=correct,
            room_state=task.room,
            agent_state=self.agent,
            evaluation_info=info,
            evaluation_data=task.eval_data
        )
        self.turn_logs.append(turn_log)
        
        return correct, info

    def next_task(self) -> bool:
        """Move to next task. Returns True if there are more tasks."""
        self.current_index += 1
        return self.current_index < len(self.tasks)
    
    def get_last_room_state(self) -> Tuple[Room, Agent]:
        """Get current room and agent state."""
        task = self.tasks[self.current_index - 1]
        return task.room, task.agent
    
    def get_eval_summary(self) -> Dict[str, Any]:
        """Calculate evaluation summary from turn logs."""
        total_tasks = len(self.tasks)
        answered_tasks = len(self.turn_logs)
        unanswered_count = total_tasks - answered_tasks
        correct_count = sum(1 for log in self.turn_logs if log.is_correct)
        incorrect_count = answered_tasks - correct_count
        
        return {
            "accuracy": correct_count / total_tasks if total_tasks > 0 else 0.0,
            "total_tasks": total_tasks,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unanswered_count": unanswered_count
        }
    
    @staticmethod
    def aggregate_group_performance(eval_summaries: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation performance for a group."""
        if not eval_summaries:
            return {"avg_accuracy": 0.0, "avg_correct_rate": 0.0, "avg_incorrect_rate": 0.0, "avg_unanswered_rate": 0.0}
        
        total_tasks = sum(m.get('total_tasks', 0) for m in eval_summaries)
        if total_tasks == 0:
            return {"avg_accuracy": 0.0, "avg_correct_rate": 0.0, "avg_incorrect_rate": 0.0, "avg_unanswered_rate": 0.0}
        
        total_correct = sum(m.get('correct_count', 0) for m in eval_summaries)
        total_incorrect = sum(m.get('incorrect_count', 0) for m in eval_summaries)
        total_unanswered = sum(m.get('unanswered_count', 0) for m in eval_summaries)
        
        return {
            "avg_accuracy": sum(m.get('accuracy', 0) for m in eval_summaries) / len(eval_summaries),
            "avg_correct_rate": total_correct / total_tasks,
            "avg_incorrect_rate": total_incorrect / total_tasks,
            "avg_unanswered_rate": total_unanswered / total_tasks
        }
    
    def reset(self):
        """Reset to start."""
        self.current_index = 0
        self.results = []
    
    def __len__(self):
        return len(self.tasks)


if __name__ == "__main__":
    pass