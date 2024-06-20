import math


class Config:
    """
    The configuration for the whole system
    """

    OBJ_NUM = 100
    KNOWN_OBJECT_NUM = 10

    LATENT_DIM = 10
    MAX_ACTION = 0.1
    POSITION_TOLERANCE = 0.04
    VISIBLE_REPRESENTATION_NOISE = 0.1
    TASK_TYPES = ["gripping", "pushing", "inserting"]

    ORACLE_COST = 1

    DEMO_NOISE = 0.002  # set like that by replaying trajectories for the same object, chosen when around 2-5% failed
    ACTION_EXPLORATION_DEVIATION = 0.5
    EXPLORATION_TRIES = 100000
    SIMILARITY_THRESHOLD = 0.85
    MIN_TRAJ_STEPS = 5
    TOP_K = 5

    VERBOSITY = 1
    METHOD = "average"

    SMOOTHING_TRIES = 10

    IMAGE_DIRECTORY = "images"

    SA_T = 29.402
    SA_T_MIN = 0.1139

    TS_L = 10000
    TS_GAMMA = 2

    RHC_ITER = 5

    ES_ELITE_PROP = 0.23
    ES_POP_SIZE = 112
    ES_MUTATION_RATE = 0.15

    PSO_PARTICLES = 110
    PSO_C1 = 2.6
    PSO_C2 = 1.5
    PSO_W = 0.99
    PSO_K = 25
    PSO_P = 2

    MP_OPTIMISER_NAME = "OriginalWarSO"

    MH_BUDGET = 10000

    THRESH_ESTIMATION_STRATEGY = "density"
    USE_REAL_THRESHOLD = False
    USE_ACTUAL_EVALUATION = False

    SEED = 0

    def __init__(self):
        pass

    def __str__(self):
        return f"Config: {self.__dict__}"
