FALSE_BELIEF_INSTRUCTION = """\
You have returned to the initial position and orientation.
There are {n_changes} objects in the room that have been changed (position or orientation).
Goal: Explore the room again and identify which objects have been changed and how.
You must use the Term(changes="...") action to submit your answer and terminate.
Format: Term(changes="object1: change_type, object2: change_type")
Example: Term(changes="apple: position, chair: orientation")
"""
