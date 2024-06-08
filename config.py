class Config:
    """
    The configuration for the whole system
    """
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

    SA_T = 50
    SA_T_MIN = 0.0001
    SA_ALPHA = 0.9
    SA_ITER = 5000

    TS_L = 10000
    TS_ITER = 5000
    TS_GAMMA = 2

    RHC_ITER = 5

    MH_TIME_BUDGET = 0.5  # in seconds

    def __init__(self):
        pass

    def __str__(self):
        return f"Config: {self.__dict__}"
