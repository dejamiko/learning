import json
import os
import re
import shutil
import time

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from optim.model_training import SiameseNetwork, generate_training_images
from playground.storage import ObjectStorage
from tm_utils import (
    ImageEmbeddings,
    ImagePreprocessing,
    NNSimilarityMeasure,
    NNImageEmbeddings,
)


def remove_embeddings():
    parent = "_data"
    for d in os.listdir(parent):
        if not os.path.isdir(os.path.join(parent, d)):
            continue
        for name in [
            "",
            "background_rem",
            "cropping, background_rem, greyscale",
            "cropping, background_rem",
            "cropping, greyscale",
            "cropping",
            "greyscale",
            "segmentation",
        ]:
            if os.path.exists(
                os.path.join(parent, d, f"embeddings_siamese, [{name}].json")
            ):
                os.remove(os.path.join(parent, d, f"embeddings_siamese, [{name}].json"))


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
        for m in NNImageEmbeddings:
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
    training_data_dir = "training_data"
    if os.path.exists(training_data_dir):
        shutil.rmtree(training_data_dir)
    df = pd.read_csv("_data/training_data/similarity_df.csv")
    for ps in processing_steps_to_try:
        config.IMAGE_PREPROCESSING = ps
        generate_training_images(ps, training_data_dir)
        for sm in NNSimilarityMeasure:
            start = time.time()
            similarities = {}
            for row in df.itertuples():
                path_1 = row.image1_path
                path_2 = row.image2_path

                similarities[f"{path_1},{path_2}"] = []

                for i in range(5):
                    a = Image.open(
                        os.path.join(training_data_dir, path_1, f"image_{i}.png")
                    )
                    b = Image.open(
                        os.path.join(training_data_dir, path_2, f"image_{i}.png")
                    )

                    if sm == NNSimilarityMeasure.TRAINED:
                        sim = get_own_trained(a, b, config, ps)
                    elif sm == NNSimilarityMeasure.FINE_TUNED:
                        sim = get_fine_tuned(a, b, config, ps)
                    elif sm == NNSimilarityMeasure.LINEARLY_PROBED:
                        sim = get_linearly_probed(a, b, config, ps)

                    similarities[f"{path_1},{path_2}"].append(sim)

            print("Done", sm, time.time() - start)

            with open(
                os.path.join(
                    "_data/siamese_similarities", f"similarities_{sm.value}_{ps}.json"
                ),
                "w",
            ) as f:
                json.dump(similarities, f)
        shutil.rmtree(training_data_dir)


def read_training_results():
    processing_steps_to_try = [
        [],
        [ImagePreprocessing.GREYSCALE.value],
        [ImagePreprocessing.BACKGROUND_REM.value],
        [ImagePreprocessing.CROPPING.value],
        [ImagePreprocessing.SEGMENTATION.value],
        [ImagePreprocessing.CROPPING.value, ImagePreprocessing.BACKGROUND_REM.value],
        [ImagePreprocessing.CROPPING.value, ImagePreprocessing.GREYSCALE.value],
        [
            ImagePreprocessing.CROPPING.value,
            ImagePreprocessing.BACKGROUND_REM.value,
            ImagePreprocessing.GREYSCALE.value,
        ],
    ]
    with open("optim/training_res.txt", "r") as f:
        all_lines = f.readlines()
    all_lines = [a.strip() for a in all_lines]
    line_ind = 0
    pattern = r"Epoch \[\d*\/200\], Train Loss: (\d*.\d*), Val Loss: (\d*.\d*)"
    losses = {}
    for ps in processing_steps_to_try:
        losses[",".join(ps)] = {}
        for m in NNSimilarityMeasure:
            validation_loss = []
            training_loss = []
            # read until early stopping
            while all_lines[line_ind] != "Early stopping":
                match = re.match(pattern, all_lines[line_ind])
                training_loss.append(float(match.group(1)))
                validation_loss.append(float(match.group(2)))
                line_ind += 1
            line_ind += 1
            losses[",".join(ps)][m.value] = {
                "training": training_loss,
                "validation": validation_loss,
            }
    with open("optim/training_res.json", "w") as f:
        json.dump(losses, f)


def generate_df():
    image1_paths = []
    image2_paths = []
    similarities = []

    with open("_data/training_data/training_objects.json", "r") as f:
        objects = json.load(f)

    with open("_data/training_data/training_transfers.json", "r") as f:
        transforms = json.load(f)

    for obj_names, sim in transforms.items():
        obj1, obj2 = obj_names.split("-")
        image1_paths.append(objects[obj1])
        image2_paths.append(objects[obj2])
        similarities.append(sim)

    df = pd.DataFrame(
        {
            "image1_path": image1_paths,
            "image2_path": image2_paths,
            "similarity": similarities,
        }
    )

    df.to_csv("_data/training_data/similarity_df.csv")


if __name__ == "__main__":
    remove_embeddings()
    calculate_all_embeddings()
