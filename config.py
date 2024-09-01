import json

import torch.cuda

from tm_utils import (
    VisualisationMethod,
    ObjectSelectionStrategyThreshold,
    SimilarityMeasure,
    ImageEmbeddings,
    ObjectSelectionStrategyAffine,
)


class Config:
    """
    The configuration for the whole system
    """

    SEED = 0
    VERBOSITY = 1

    USE_REAL_OBJECTS = True

    SUCCESS_RATE_BOOLEAN = True
    SIMILARITY_THRESHOLDS = [0.55, 0.55, 0.55]
    PROB_THRESHOLD = 0.70  # this should be treated as a constant

    VISUALISATION_METHOD = VisualisationMethod.PCA

    DEMONSTRATION_BUDGET = 5

    SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_FULL
    USE_ALL_IMAGES = False
    IMAGE_PREPROCESSING = []

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE = "dino_vits8"
    STRIDE = 8
    LOAD_SIZE = 256
    FACET = "key"
    BIN = True

    MASK_RCNN_THRESHOLD = 0.003
    PANOPTIC_FPN_THRESHOLD = 0.004
    CASCADE_MASK_RCNN_THRESHOLD = 0.003

    # Threshold approximation solver
    OBJECT_SELECTION_STRATEGY_T = ObjectSelectionStrategyThreshold.INTERVALS
    USE_TRANSFER_EVALUATION = False

    # Affine approximation solver
    MERGING_FACTOR = 0.5
    OBJECT_SELECTION_STRATEGY_A = ObjectSelectionStrategyAffine.GREEDY_R

    DO_NOT_ITER = False

    # Random object
    OBJ_NUM = 40
    LATENT_DIM = 10
    VISIBLE_REPRESENTATION_NOISE = 0.1

    # Metaheuristic-related
    SA_T = 29.402
    SA_T_MIN = 0.1139

    TS_L = 10000

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

    MH_BUDGET = 2000
    MH_TIME_BUDGET = 1  # in seconds

    def __init__(self):
        pass

    def __str__(self):  # pragma: no cover
        data = self.__dict__.copy()
        if "VISUALISATION_METHOD" in data:
            data["VISUALISATION_METHOD"] = data["VISUALISATION_METHOD"].value
        if "SIMILARITY_MEASURE" in data:
            data["SIMILARITY_MEASURE"] = data["SIMILARITY_MEASURE"].value
        if "IMAGE_EMBEDDINGS" in data:
            data["IMAGE_EMBEDDINGS"] = data["IMAGE_EMBEDDINGS"].value
        if "OBJECT_SELECTION_STRATEGY_T" in data:
            data["OBJECT_SELECTION_STRATEGY_T"] = data[
                "OBJECT_SELECTION_STRATEGY_T"
            ].value
        if "OBJECT_SELECTION_STRATEGY_A" in data:
            data["OBJECT_SELECTION_STRATEGY_A"] = data[
                "OBJECT_SELECTION_STRATEGY_A"
            ].value
        if "IMAGE_PREPROCESSING" in data:
            data["IMAGE_PREPROCESSING"] = [a.value for a in data["IMAGE_PREPROCESSING"]]
        return f"Config: {json.dumps(data)}"

    def get_embedding_spec(self):
        return f"{self.IMAGE_EMBEDDINGS.value}, [{', '.join([a.value for a in self.IMAGE_PREPROCESSING])}]"
