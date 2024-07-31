from utils import (
    SingletonMeta,
    VisualisationMethod,
    ThresholdEstimationStrategy,
    SimilarityMeasure,
)


class Config(metaclass=SingletonMeta):
    """
    The configuration for the whole system
    """

    SEED = 0
    VERBOSITY = 1

    USE_REAL_OBJECTS = True

    SUCCESS_RATE_BOOLEAN = True
    SIMILARITY_THRESHOLD = 0.55
    PROB_THRESHOLD = 0.70  # this should be treated as a constant

    VISUALISATION_METHOD = VisualisationMethod.PCA.value

    DEMONSTRATION_BUDGET = 5

    SIMILARITY_MEASURE = SimilarityMeasure.DINO_LAYER_9_COSINE.value

    # Variable threshold solver
    THRESH_ESTIMATION_STRATEGY = ThresholdEstimationStrategy.DENSITY.value
    USE_REAL_THRESHOLD = False
    USE_TRANSFER_EVALUATION = False

    # Random object
    OBJ_NUM = 100
    LATENT_DIM = 10
    VISIBLE_REPRESENTATION_NOISE = 0.1

    # Metaheuristic-related
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

    MH_BUDGET = 45000
    MH_TIME_BUDGET = 0.8  # in seconds

    def __init__(self):
        pass

    def __str__(self):  # pragma: no cover
        return f"Config: {self.__dict__}"
