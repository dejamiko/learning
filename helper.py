import json
import os
import time

from config import Config
from playground.environment import Environment
from tm_utils import ImageEmbeddings, ImagePreprocessing


def remove_embeddings():
    parent = "_data"
    for d in os.listdir(parent):
        if not os.path.isdir(os.path.join(parent, d)):
            continue
        if os.path.exists(os.path.join(parent, d, "embeddings.json")):
            os.remove(os.path.join(parent, d, "embeddings.json"))


def embedding_change():
    parent = "_data"
    for d in os.listdir(parent):
        if not os.path.isdir(os.path.join(parent, d)):
            continue
        with open(os.path.join(parent, d, "embeddings.json"), "r") as f:
            data = json.load(f)
        new_data = {}
        for im_dict in data:
            for k in im_dict:
                if k not in new_data:
                    new_data[k] = []
                new_data[k].append(im_dict[k])
        for k, v in new_data.items():
            with open(os.path.join(parent, d, f"embeddings_{k}.json"), "w") as f:
                json.dump(v, f)
        os.remove(os.path.join(parent, d, "embeddings.json"))


def calculate_all_embeddings():
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
    for ps in processing_steps_to_try:
        config.IMAGE_PREPROCESSING = ps
        for m in ImageEmbeddings:
            start_time = time.time()
            config.IMAGE_EMBEDDINGS = m
            _ = Environment(config)
            print(f"Method {m.value} done in {time.time() - start_time} s")
        print(f"Preprocessing steps done {ps}")


if __name__ == "__main__":
    pass
