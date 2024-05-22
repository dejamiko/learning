OBJ_NUM = 50
KNOWN_OBJECT_NUM = 10

LATENT_DIM = 2
MAX_ACTION = 0.1
POSITION_TOLERANCE = 0.04
VISIBLE_REPRESENTATION_NOISE = 0.1
TASK_TYPES = [
    "gripping",
    "pushing",
    "inserting"
]

ORACLE_COST = 1

DEMO_NOISE = 0.001  # set like that by replaying trajectories for the same object, chosen when around 2-5% failed
ACTION_EXPLORATION_DEVIATION = 0.5
EXPLORATION_TRIES = 100000
SIMILARITY_THRESHOLD = 0.999

VERBOSITY = 1
