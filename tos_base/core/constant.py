import numpy as np
from .object import Object


AGENT_NAME = 'you'

CANDIDATE_OBJECTS = [
    'table',
    'chair',
    'sofa',
    'bed',
    'desk',
    'bookshelf',
    'cabinet',
    'lamp',
    'television',
    'refrigerator',
    'microwave',
    'oven',
    'toaster',
    'clock',
    'plant',
    'flower',
    'vase',
    'computer',
    'printer',
    'scanner',
    'monitor',
    'keyboard',
    'mouse',
    'headphones',
    'microphone',
    'projector',
    'whiteboard',
    'notebook',
    'pen',
    'pencil',
    'eraser',
    'stapler',
    'folder',
]


ADDITIONAL_CANDIDATE_OBJECTS = [
    'curtains',
    'mirror',
    'rug',
    'pillow',
    'blanket',
    'dresser',
    'nightstand',
    'fan',
    'air conditioner',
    'heater',
    'trash can',
    'picture frame',
    'wall art',
    'throw pillow',
    'coffee table',
    'ottoman',
    'ceiling light',
    'phone charger',
    'coaster',
    'decorative bowl'
]



# predefined room
easy_room_config = {
    "name": "easy_room",
    "objects": [
        Object(
            name="table",
            pos=np.array([3, 0]),
            ori=np.array([1, 0])),
        Object(
            name="chair",
            pos=np.array([3, 3]),
            ori=np.array([-1, 0])),
    ],
    "Agent": Object(
        name="agent",
        pos=np.array([0, 0]),
        ori=np.array([0, 1])
    ),
}

easy_room_config_2 = {
    "name": "easy_room_2",
    "objects": [
        Object(
            name="table", 
            pos=np.array([4, -3]),
            ori=np.array([1, 0])),
        Object(
            name="chair", 
            pos=np.array([3, 3]),
            ori=np.array([0, -1])),
        Object(
            name="sofa", 
            pos=np.array([2, 0]),
            ori=np.array([-1, 0])),
    ],
    "Agent": Object(
        name="agent", 
        pos=np.array([0, 0]), 
        ori=np.array([0, 1])),
}


# easy_room_config_3 = {
#     "name": "easy_room_3",
#     "objects": [
#         Object(
#             name="table", 
#             pos=np.array([4, -3]),
#             ori=np.array([1, 0])),
#         Object(
#             name="chair", 
#             pos=np.array([3, 3]),
#             ori=np.array([0, -1])),
#         Object(
#             name="sofa", 
#             pos=np.array([2, 0]),
#             ori=np.array([-1, 0])),
#     ],
# }

easy_room_config_3 = {
    "name": "easy_room_3",
    "objects": [
        Object(
            name="A", 
            pos=np.array([0, 0]),
            ori=np.array([1, 0])),
        Object(
            name="B", 
            pos=np.array([1, 0]),
            ori=np.array([0, -1])),
        Object(
            name="C", 
            pos=np.array([0, -1]),
            ori=np.array([-1, 0])),
        Object(
            name="D", 
            pos=np.array([-1, 0]),
            ori=np.array([0, 1])),
        Object(
            name="E", 
            pos=np.array([0, 1]),
            ori=np.array([0, 1])),
    ],
}