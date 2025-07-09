import numpy as np
from .tasks import (
    BaseEvaluationTask,
    DirEvaluationTask,
    RotEvaluationTask,
    AllPairsEvaluationTask,
    ReverseDirEvaluationTask,
    PovEvaluationTask,
    E2AEvaluationTask,
)

def get_eval_task(eval_task: str, np_random: np.random.Generator, eval_kwargs: dict = None) -> BaseEvaluationTask:
    """
    Get the evaluation task from the config
    """
    task_map = {
        "dir": DirEvaluationTask,
        "rot": RotEvaluationTask,
        "all_pairs": AllPairsEvaluationTask,
        "rev": ReverseDirEvaluationTask,
        "pov": PovEvaluationTask,
        "e2a": E2AEvaluationTask,
    }
    
    if eval_task in task_map:
        task_class = task_map[eval_task]
        return task_class(np_random, eval_kwargs or {})
    else:
        raise ValueError(f"Unknown evaluation task: {eval_task}")
