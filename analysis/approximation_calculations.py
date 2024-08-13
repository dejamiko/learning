import json
import time

import numpy as np
from scipy.stats import linregress
from sklearn.metrics import f1_score

from analysis.similarity_measure_eval import extract_features
from config import Config
from playground.environment import Environment
from tm_utils import ImagePreprocessing, ImageEmbeddings, SimilarityMeasure, ContourImageEmbeddings, \
    ContourSimilarityMeasure, NNImageEmbeddings, NNSimilarityMeasure


def convert_binary(arr, threshold):
    return np.where(arr >= threshold, 1, 0)


def get_f1_score(a1, a2):
    return f1_score(a1.flatten(), a2.flatten())


def get_threshold(ls, vs, config):
    # find a threshold, perhaps through exhaustive search, that maximises the correct boolean conversion
    # between ls and vs, where the sim thresh for ls is >= 0.7
    ls_bin = convert_binary(ls, config.PROB_THRESHOLD)
    best_threshold = None
    best_f1 = None
    for threshold in np.linspace(0.0, 1.0, 200):
        vs_bin = convert_binary(vs, threshold)
        f1_score = get_f1_score(ls_bin, vs_bin)

        if best_f1 is None:
            best_f1 = f1_score
            best_threshold = threshold
        elif best_f1 < f1_score:
            best_f1 = f1_score
            best_threshold = threshold

    return best_threshold


def get_affine_fn(ls, vs):
    slope, intercept, r, p, std_err = linregress(vs.flatten(), ls.flatten())
    return float(slope), float(intercept)


def run_one(config):
    start = time.time()
    environment = Environment(config)
    ls_all, vs_all, tasks = extract_features(environment, config)
    thresholds = []
    affine_fns = []
    for ls, vs, task in zip(ls_all, vs_all, tasks):
        threshold = get_threshold(ls, vs, config)
        affine_fn = get_affine_fn(ls, vs)
        thresholds.append(threshold)
        affine_fns.append(affine_fn)

    print(f"Finished {str(config)} in {time.time() - start}")
    return thresholds, affine_fns


if __name__ == "__main__":
    config = Config()
    config.OBJ_NUM = 51
    processing_steps_to_try = [
        [],
        [ImagePreprocessing.GREYSCALE],
        [ImagePreprocessing.BACKGROUND_REM],
        [ImagePreprocessing.CROPPING],
        [ImagePreprocessing.SEGMENTATION],
        [ImagePreprocessing.CROPPING, ImagePreprocessing.BACKGROUND_REM],
        [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
        [
            ImagePreprocessing.CROPPING,
            ImagePreprocessing.BACKGROUND_REM,
            ImagePreprocessing.GREYSCALE,
        ],
    ]
    data = {}
    for ps in processing_steps_to_try:
        config.IMAGE_PREPROCESSING = ps
        for emb in ImageEmbeddings:
            config.IMAGE_EMBEDDINGS = emb
            for sim in SimilarityMeasure:
                config.SIMILARITY_MEASURE = sim
                thresholds, affine_fns = run_one(config)
                data[str(config)] = {"thresholds": thresholds, "affine_fns": affine_fns}

        for emb in ContourImageEmbeddings:
            config.IMAGE_EMBEDDINGS = emb
            for sim in ContourSimilarityMeasure:
                config.SIMILARITY_MEASURE = sim
                thresholds, affine_fns = run_one(config)
                data[str(config)] = {"thresholds": thresholds, "affine_fns": affine_fns}

        for emb in NNImageEmbeddings:
            config.IMAGE_EMBEDDINGS = emb
            for sim in NNSimilarityMeasure:
                config.SIMILARITY_MEASURE = sim
                thresholds, affine_fns = run_one(config)
                data[str(config)] = {"thresholds": thresholds, "affine_fns": affine_fns}

    with open("analysis/approximations.json", "w") as f:
        json.dump(data, f)
