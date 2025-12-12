GOAL_EXPLORATION = "Goal: Build a **COMPLETE AND ACCURATE MAP** of the environment with **MINIMAL TOTAL COST**."

ENV_RULES_HEADER = (
    "- Rooms connect via doors\n"
    "- Vision: 90° FOV. Confined to current room. No occlusion in room. "
    "Doors block vision. Exception: When located in a doorway, door is open and invisible, you can see into both connected rooms."
)

VISION_EXAMPLE = """\
Here is an example of your observation: blue cylinder 1 m straight ahead; red cylinder 2 m straight ahead; yellow cylinder 2 m at 45° to your front-left; green cylinder 3 m at 22.5° to your front-slight-right:
{image_placeholder}

The image shows all objects in the room. Each tile is numbered (1-N) in the top-left, matching the object order in the room layout.
For items with a facing direction, two copies are shown side-by-side: the left copy has its front facing the camera; the right copy has its front facing left.
Items without a meaningful facing direction are shown once.
{image_placeholder}
"""

INSTRUCTION_TEMPLATE_TEXT = """\
Role: Spatial Reasoner in a 2D N×M grid.
{goal_lines}

## Environment Rules
{env_rules_header}
{observation_instructions}

## Actions & Grammar
{exp_instructions}

## Current Context:
{room_info}
{context_footer}
{exp_history}

{format_rules}
"""

INSTRUCTION_TEMPLATE_VISION = """\
Role: Spatial Reasoner in a 2D N×M grid.
{goal_lines}

## Environment Rules
{env_rules_header}
{observation_instructions}

## Actions & Grammar
{exp_instructions}

## Current Context:
{room_info}
{context_footer}

{vision_example}
{exp_history}

{format_rules}
"""

EVALUATION_INSTRUCTION = "{eval_question}"