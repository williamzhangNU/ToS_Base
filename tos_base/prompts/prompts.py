ACTIVE_INSTRUCTION_TEXT = """\
# Spatial Exploration

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

Goal: Your objective is to **minimize total COST** while gaining knowledge of spatial relationships between each pair of objects.

Multi-room rules:
- You cannot see objects in other rooms.
- You cannot see through a door unless you are standing on it. When at a door, it's open and invisible.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
- When standing on a door, you can see objects from both connected rooms (within your FOV).

Observation:
You egocentric observation rule is provided in following format:
{observation_instructions}

Rules:
- Achieve complete coverage with the fewest steps;
- Prefer actions that reveal more unknowns; avoid redundancy
- FOV is 90°, you can NOT see objects outside your FOV.
- Track your current and initial pose

Room Layout:
{room_info}

Action Instructions:
{exp_instructions}
"""


PASSIVE_INSTRUCTION_TEXT = """\
# Spatial Understanding Task

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

Multi-room rules:
- You cannot see objects in other rooms.
- You cannot see through a door unless you are standing on it. When at a door, it's open and invisible.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
- When standing on a door, you can see objects from both connected rooms (within your FOV).

Observation:
You egocentric observation rule is provided in following format:
{observation_instructions}

Rules:
- FOV is 90°, you can NOT see objects outside your FOV.
- Track your current and initial pose

Room Layout:
{room_info}

{exp_history}
"""

ACTIVE_INSTRUCTION_VISION = """\
# Spatial Exploration Task

Goal: Your objective is to **minimize total COST** while gaining knowledge of spatial relationships between each pair of objects. Do NOT stop until you have aciehved complete coverage

!!! IMPORTANT OUTPUT RULES !!!

1. You must always output in this format:
   <answer> Actions: [ ... ] </answer>

2. Inside <answer>, ONLY output the "Actions: [...]" list exactly as specified.
   - No bullet points
   - No prose
   - No cost calculations
   - No boxed answers
   - No explanations

3. If you write anything outside <answer>, it is WRONG and you will be penalized.

Action Instructions:
{exp_instructions}

Room Layout:
{room_info}

Facing: forward/backward/right/left. When facing north: forward=north, back=south, right=east, left=west.

Multi-room: 
- Rooms are connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. When you stand at a door, you can see objects from both connected rooms (within FOV).

Observation: For visible objects you receive (direction, signed degree, distance).
- direction uses <vertical>-<horizontal> with front|back|same and left|right|same and
- degree is clockwise from your facing; distance is Euclidean
- You may ignore degree/distance for the stopping condition

Rules:
- Achieve complete coverage with the fewest steps; continue only while any pair is unknown
- Prefer actions that reveal many unknowns; avoid redundancy
- FOV is 90°
- Track your current and initial pose

Here is an example of your observation: blue object 1 m straight ahead; yellow object 2 m at 45° to your left; green object 3 m at 22.5° to your right:
{image_placeholder}

All objects in the following image are facing towards the camera and the labels match the objects listed below.
{image_placeholder}

After exploration, you will return to your starting position facing north.
"""

PASSIVE_INSTRUCTION_VISION = """\
# Spatial Understanding Task

You will be given a multi-room layout and a tour (you return to start). Then answer the question.

Facing
- forward, backward, right, left. When facing north: forward=north, back=south, right=east, left=west.

Multi-room: 
- Rooms are connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. When you stand at a door, you can see objects from both connected rooms (within FOV).

Here is an example of your observation: blue object 1 m straight ahead; yellow object 2 m at 45° to your left; green object 3 m at 22.5° to your right:
{image_placeholder}

All objects in the following image are facing towards the camera and the labels match the objects listed below.
{image_placeholder}

## Room Layout
{room_info}

{exp_history}
"""

# NOTE: COGNITION_MAP_INSTRUCTION has been moved to CognitiveMap class for flexible formatting
# The dynamic instruction is now provided by CognitiveMap.get_json_format_instruction()

EVALUATION_INSTRUCTION = "{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."