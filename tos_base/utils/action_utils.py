"""
Utility functions for converting action results to text observations.
"""
from typing import List
from ..actions import ActionResult


def action_results_to_text(action_results: List[ActionResult]) -> str:
    """Convert list of ActionResults to text observation.
    
    Args:
        action_results: List of ActionResult objects from action execution
    
    Returns:
        Text observation string
    """
    assert action_results, "action_results is empty"
    return " ".join([result.message for result in action_results])