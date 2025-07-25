"""Simple Evaluation Manager for SpatialGym Environment"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..evaluation.task_types import EvalTaskType
from ..core.room import Room
from ..evaluation.tasks import BaseEvaluationTask


class EvaluationManager:
    """
    Manages evaluation tasks for the SpatialGym environment.
    
    Handles task initialization, question generation, answer evaluation,
    and tracking of evaluation results across multiple tasks.

    TODO handle unanswerer question
    """
    
    def __init__(self, eval_tasks: List[Dict[str, Any]], np_random: np.random.Generator):
        self.eval_tasks = eval_tasks
        self.np_random = np_random
        self.results = []
        
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
        self.eval_metrics_log: List[Dict[str,Any]] = []
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
        correct_answer = task.answer
        self.eval_metrics_log.append(correct_answer)
        return correct, info
    def get_eval_metrics_log(self) -> List[Dict[str,Any]]:
        return self.eval_metrics_log
    def next_task(self) -> bool:
        """Move to next task. Returns True if there are more tasks."""
        self.current_index += 1
        return self.current_index < len(self.tasks)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        total_tasks = len(self.tasks)
        correct_count = sum(1 for r in self.results if r["correct"])
        
        return {
            "total_tasks": total_tasks,
            "accuracy": correct_count / total_tasks if total_tasks > 0 else 0.0,
            "task_results": self.results
        }
    
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
        perspective="ego",
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