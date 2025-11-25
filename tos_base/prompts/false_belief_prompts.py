FALSE_BELIEF_INSTRUCTION = """\
Action space is the same as previous exploration.
Goal: Find the {target_object}.
Terminate as soon as you observe the {target_object}.
After termination, your final viewpoint will be checked to see if target is in it.
You return to your initial position and orientation.
"""
