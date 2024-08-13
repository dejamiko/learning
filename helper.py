import json
import os
import shutil
import time

import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from optim.model_training import SiameseNetwork, generate_training_images
from playground.extractor import Extractor
from playground.storage import ObjectStorage
from tm_utils import ImageEmbeddings, ImagePreprocessing, NNSimilarityMeasure


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
            storage = ObjectStorage(config)
            storage.generate_objects()
            print(f"Method {m.value} done in {time.time() - start_time} s")
        print(f"Preprocessing steps done {ps}")


def get_own_trained(a, b, config, ps):
    return get_nn_sim(
        a,
        b,
        {"frozen": False, "backbone": False},
        f"models/siamese_net_trained_{ps}.pth",
        config,
    )


def get_fine_tuned(a, b, config, ps):
    return get_nn_sim(
        a,
        b,
        {"frozen": True, "backbone": True},
        f"models/siamese_net_fine_tuned_{ps}.pth",
        config,
    )


def get_linearly_probed(a, b, config, ps):
    return get_nn_sim(
        a,
        b,
        {"frozen": False, "backbone": True},
        f"models/siamese_net_linearly_probed_{ps}.pth",
        config,
    )


def get_nn_sim(a, b, model_config, model_path, config):
    transform = transforms.Compose(
        [
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    a = transform(a).unsqueeze(0).to(config.DEVICE)
    b = transform(b).unsqueeze(0).to(config.DEVICE)
    model = SiameseNetwork(**model_config)
    model.load_state_dict(
        torch.load(os.path.join("optim", model_path), map_location=torch.device("cpu"))
    )
    model.to(config.DEVICE)
    model.eval()
    res = float(model(a, b).squeeze().detach().cpu().numpy())
    return min(max(res, 0.0), 1.0)


def calculate_own_models_sim():
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
    df = pd.read_csv("_data/similarity_df.csv")
    for ps in processing_steps_to_try:
        config.IMAGE_PREPROCESSING = ps
        generate_training_images(ps)
        for sm in NNSimilarityMeasure:
            start = time.time()
            similarities = {}
            for row in df.itertuples():
                path_1 = row.image1_path
                path_2 = row.image2_path

                a = cv2.imread(os.path.join(path_1, "image_0.png"))
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                a = Extractor._apply_preprocessing(
                    config, a, os.path.join(path_1, "image_0.png")
                )
                a = Image.fromarray(a)

                b = cv2.imread(os.path.join(path_2, "image_0.png"))
                b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
                b = Extractor._apply_preprocessing(
                    config, b, os.path.join(path_2, "image_0.png")
                )
                b = Image.fromarray(b)

                if sm == NNSimilarityMeasure.TRAINED:
                    sim = get_own_trained(a, b, config, ps)
                elif sm == NNSimilarityMeasure.FINE_TUNED:
                    sim = get_fine_tuned(a, b, config, ps)
                elif sm == NNSimilarityMeasure.LINEARLY_PROBED:
                    sim = get_linearly_probed(a, b, config, ps)

                similarities[f"{path_1},{path_2}"] = sim
            print("Done", sm, time.time() - start)

            with open(
                os.path.join("_data", f"similarities_{sm.value}_{ps}.json"), "w"
            ) as f:
                json.dump(similarities, f)
        shutil.rmtree("training_data")


if __name__ == "__main__":
    calculate_own_models_sim()
