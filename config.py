class Config:
    """
    The configuration for the whole system
    """

    OBJ_NUM = 100
    KNOWN_OBJECT_NUM = 10

    LATENT_DIM = 10
    VISIBLE_REPRESENTATION_NOISE = 0.1
    TASK_TYPES = ["gripping", "pushing", "inserting"]

    SIMILARITY_THRESHOLD = 0.85

    VERBOSITY = 1

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
    MH_TIME_BUDGET = 0.8  # in seconds

    THRESH_ESTIMATION_STRATEGY = "density"
    USE_REAL_THRESHOLD = False
    USE_TRANSFER_EVALUATION = False

    SEED = 0

    VISUALISATION_METHOD = "pca"

    def __init__(self):
        pass

    def __str__(self):
        return f"Config: {self.__dict__}"
