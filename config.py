from tm_utils import (
    VisualisationMethod,
    ObjectSelectionStrategy,
    SimilarityMeasure,
    ImageEmbeddings,
)


class Config:
    """
    The configuration for the whole system
    """

    SEED = 0
    VERBOSITY = 1

    USE_REAL_OBJECTS = True

    SUCCESS_RATE_BOOLEAN = True
    SIMILARITY_THRESHOLD = 0.55
    PROB_THRESHOLD = 0.70  # this should be treated as a constant

    VISUALISATION_METHOD = VisualisationMethod.PCA

    DEMONSTRATION_BUDGET = 5

    SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    SIM_MEASURE_SIGMA = 1.0
    IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_2_FULL

    DEVICE = "cpu"
    MODEL_TYPE = "dino_vits8"
    STRIDE = 8
    LOAD_SIZE = 256
    FACET = "key"
    BIN = True

    # Threshold approximation solver
    OBJECT_SELECTION_STRATEGY = ObjectSelectionStrategy.DENSITY
    USE_REAL_THRESHOLD = False
    USE_TRANSFER_EVALUATION = False

    # Affine approximation solver
    MERGING_FACTOR = 0.5

    # Random object
    OBJ_NUM = 40
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
    MH_TIME_BUDGET = 8  # in seconds

    def __init__(self):
        pass

    def __str__(self):  # pragma: no cover
        return f"Config: {self.__dict__}"
