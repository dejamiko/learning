import json
import os
import time

from config import Config
from playground.environment import Environment
from tm_utils import ImageEmbeddings, ImagePreprocessing

if __name__ == "__main__":
    config = Config()
    config.OBJ_NUM = 51
    config.IMAGE_PREPROCESSING = ImagePreprocessing.GREYSCALE
    for m in ImageEmbeddings:
        start_time = time.time()
        config.IMAGE_EMBEDDINGS = m
        env = Environment(config)
        print(f"Method {m.value} done in {time.time() - start_time} s")
