SHARED_INTRO_TEXT = "You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates."
SHARED_INTRO_VISION = (
    "You are a spatial reasoner in a 3D simulated environment. "
    "The world is rendered in 3D but abstracted into a discrete 2D grid of size N×M. "
    "Every entity, including yourself, is represented by integer coordinates (x, y) on this grid."
)

SHARED_MULTIROOM_RULES = """\
Multi-room rules (may exist multiple rooms):
- Your vision is confined to your current room.
- Doors block vision between rooms.
- Exception: When located in a doorway, door is open and invisible, you can see into both connected rooms.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
"""

SHARED_RULES_COMMON = """\
- FOV is 90°, you can NOT see objects outside your FOV.
- Track your current and initial pose
"""

ACTIVE_RULES_EXTRA = """\
- Achieve complete coverage with the fewest steps;
- Prefer actions that reveal more unknowns; avoid redundancy
"""

VISION_EXAMPLE = """\
Here is an example of your observation: blue object 1 m straight ahead; yellow object 2 m at 45° to your left; green object 3 m at 22.5° to your right:
{image_placeholder}

All objects in the following image are facing towards the camera (facing backwards). And numbered from 1 to N corresponding to object sequence mentioned in the room layout.
{image_placeholder}
"""

INSTRUCTION_TEMPLATE_TEXT = """\
# {title}

{intro}

{goal_lines}

{multiroom_rules}

Relationship instructions:
{observation_instructions}

Action Instructions:
{exp_instructions}

{format_rules}

Rules:
{active_rules_extra}{rules_common}

Room Layout and initial state:
{room_info}

{exp_history}
"""





INSTRUCTION_TEMPLATE_VISION = """\
# {title}

{intro}

{goal_lines}

{multiroom_rules}

Relationship instructions:
{observation_instructions}

Action Instructions:
{exp_instructions}

{format_rules}

Rules:
{active_rules_extra}{rules_common}

Room Layout and initial state:
{room_info}

{vision_example}

{exp_history}
"""

EVALUATION_INSTRUCTION = "{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."