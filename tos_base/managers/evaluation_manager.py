"""Simple Evaluation Manager for SpatialGym Environment"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..evaluation.task_types import EvalTaskType
from ..core.room import Room
from ..evaluation.tasks import BaseEvaluationTask

@dataclass
class EvaluationTurnLog:
    """Log data for a single evaluation turn."""
    task_type: str
    question: str
    user_answer: str
    correct_answer: Any
    is_correct: bool
    evaluation_info: Dict[str, Any]


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
    
    def __init__(self, eval_tasks: List[Dict[str, Any]], np_random: np.random.Generator):
        self.eval_tasks = eval_tasks
        self.np_random = np_random
        self.results = []
        self.turn_logs: List[EvaluationTurnLog] = []
        
        # Initialize tasks
        self.tasks = []
        for task_spec in eval_tasks:
            task_type = task_spec['task_type']
            task_kwargs = task_spec.get('task_kwargs', {})
            task = EvalTaskType.create_task(task_type, np_random, task_kwargs)
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
    
    def get_current_question(self, room: Room) -> Optional[str]:
        """Get question for current task."""
        task = self._get_current_eval_task()
        return None if task is None else task.question if task.question else task.generate_question(room.copy())
    
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
            question=task.question or "",
            user_answer=answer,
            correct_answer=task.answer,
            is_correct=correct,
            evaluation_info=info
        )
        self.turn_logs.append(turn_log)
        
        return correct, info
    def get_turn_logs(self) -> List[EvaluationTurnLog]:
        """Get evaluation turn logs."""
        return self.turn_logs
    
    def _calculate_evaluation_summary(self) -> Dict[str, Any]:
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

    def next_task(self) -> bool:
        """Move to next task. Returns True if there are more tasks."""
        self.current_index += 1
        return self.current_index < len(self.tasks)
    
    def get_eval_summary(self) -> Dict[str, Any]:
        """Get evaluation summary (legacy method)."""
        return self._calculate_evaluation_summary()
    
    def reset(self):
        """Reset to start."""
        self.current_index = 0
        self.results = []
    
    def __len__(self):
        return len(self.tasks)


if __name__ == "__main__":
    # Simple test
    from ..utils.room_utils import generate_room
    from ..core.constant import CANDIDATE_OBJECTS
    from gymnasium.utils import seeding
    
    eval_tasks = [{"task_type": "rot", "task_kwargs": {}}]
    np_random = seeding.np_random(42)[0]
    
    eval_manager = EvaluationManager(eval_tasks, np_random)
    room = generate_room(
        np_random=np_random,
        n_objects=3, 
        candidate_objects=CANDIDATE_OBJECTS,
        generation_type="rand",
        room_range=[-10, 10],
    )
    print(f"Room: {room}")
    
    question = eval_manager.get_current_question(room)
    print(f"Question: {question}")
    
    task = eval_manager._get_current_eval_task()
    print(f"Task Answer: {task.answer}")
    
    correct, info = eval_manager.evaluate_answer("['keyboard', 'sofa', 'microphone']")
    print(f"Result: correct={correct}, info={info}")
    
    summary = eval_manager.get_evaluation_summary()
    print(f"Summary: {summary['accuracy']}") 