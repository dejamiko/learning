import json
import os
import re
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from analysis.similarity_measure_eval import extract_features
from config import Config
from optim.model_training import SiameseNetwork, generate_training_images
from playground.environment import Environment
from playground.extractor import Extractor
from playground.storage import ObjectStorage
from tm_utils import (
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
        # for m in ImageEmbeddings:
        #     start_time = time.time()
        #     config.IMAGE_EMBEDDINGS = m
        #     storage = ObjectStorage(config)
        #     storage.generate_objects()
        #     print(f"Method {m.value} done in {time.time() - start_time} s")
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


def generate_training_plots():
    with open("optim/training_res.json", "r") as f:
        data = json.load(f)
    for emb, losses_by_model in data.items():
        for model_type, losses in losses_by_model.items():
            training = losses["training"]
            validation = losses["validation"]
            plt.close()
            plt.plot(training, label="training")
            plt.plot(validation, label="validation")
            plt.legend()
            plt.title(f"Preprocessing [{emb}] and model type `{model_type}`")
            plt.savefig(f"{emb}-{model_type}")


def get_demo_images():
    output_image = "combined_image_with_names.png"  # Final output image
    rows, cols = 9, 6  # Grid size
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font for text
    font_scale = 0.5  # Font scale
    font_thickness = 1  # Font thickness
    text_color = (0, 0, 0)

    image_files = []
    object_names = []

    parent = "_data"
    for d in sorted(os.listdir(parent)):
        if not os.path.isdir(os.path.join(parent, d)):
            continue
        if d in ["siamese_similarities", "training_data"]:
            continue
        image_files.append(os.path.join(parent, d, "image_0.png"))
        object_names.append(d)

    # Ensure you have 51 images and names
    print(len(image_files))

    # Load images into a list
    images = [cv2.imread(img) for img in image_files]

    # Resize images if necessary (optional)
    # You might want to resize to a consistent size, e.g., 100x100 pixels
    resized_images = [cv2.resize(img, (250, 250)) for img in images]

    # Create a blank canvas for the grid with space for names
    image_height = 270  # 100 pixels for image + 20 pixels for name text
    grid_height = rows * image_height
    grid_width = cols * 250
    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Place each image and its corresponding name on the canvas
    for i, (img, name) in enumerate(zip(resized_images, object_names)):
        row = i // cols
        col = i % cols
        start_y = row * image_height
        start_x = col * 250

        # Place the image
        canvas[start_y : start_y + 250, start_x : start_x + 250] = img

        # Calculate the position for the text
        text_y = start_y + 264  # Slightly below the image
        text_x = start_x + 10  # Some padding from the left

        # Add the name below the image
        cv2.putText(
            canvas,
            name,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Save the final image
    cv2.imwrite(output_image, canvas)

    # Optionally display the final image
    cv2.imshow("Combined Image with Names", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def numpy_to_latex_table(
    array, row_names=None, col_names=None, column_format=None, caption=None, label=None
):
    """
    Converts a NumPy array to a LaTeX tabular format with optional row and column names.

    Parameters:
    array (numpy.ndarray): The input NumPy array to be converted.
    row_names (list of str): List of names for the rows.
    col_names (list of str): List of names for the columns.
    column_format (str): LaTeX column format (e.g., "c", "l", "r", or combinations like "ccc").
                         If None, it defaults to centered columns.
    caption (str): Optional caption for the table.
    label (str): Optional label for referencing the table in LaTeX.

    Returns:
    str: The LaTeX string representing the table.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    num_rows, num_cols = array.shape

    # Validate row_names and col_names
    if row_names and len(row_names) != num_rows:
        raise ValueError(
            "Length of row_names must match the number of rows in the array."
        )
    if col_names and len(col_names) != num_cols:
        raise ValueError(
            "Length of col_names must match the number of columns in the array."
        )

    # Set default column format if not provided
    if column_format is None:
        column_format = (
            "|" + "|".join(["c"] * (num_cols + (1 if row_names else 0))) + "|"
        )

    # Initialize LaTeX table string
    latex_str = (
        "\\begin{table}[h]\n\\centering\n\\begin{tabular}{"
        + column_format
        + "}\n\\hline\n"
    )

    # Convert each row to a string and join them
    rows = []
    for i, row in enumerate(array):
        row_str = " & ".join(map(str, row))
        if row_names:
            row_str = row_names[i] + " & " + row_str  # Prepend row name
        rows.append(row_str)

    # Join all rows with LaTeX new line and hline for row separation
    latex_str += " \\\\\n\\hline\n".join(rows)

    # End LaTeX tabular and add optional caption and label
    latex_str += "\n\\end{tabular}\n"

    if caption:
        latex_str += f"\\caption{{{caption}}}\n"
    if label:
        latex_str += f"\\label{{{label}}}\n"

    latex_str += "\\end{table}"

    return latex_str


def print_similarity_matrix():
    c = Config()
    c.OBJ_NUM = 51
    env = Environment(c)
    ls_all, _, tasks = extract_features(env, c)
    objs = env.get_objects()
    objs_all = []
    for task in tasks:
        obj = []
        for o in objs:
            if o.task == task:
                obj.append(o.name)
        objs_all.append(obj)
    for ls, task, obj_names in zip(ls_all, tasks, objs_all):
        print(task)
        print("=" * 50)
        print(numpy_to_latex_table(ls, row_names=obj_names, col_names=obj_names))
        print("=" * 50)


def get_preprocessed_images():
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
        config = Config()
        config.IMAGE_PREPROCESSING = ps
        img = Extractor._load_images("_data/scissors_grasping", config)[0][0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"img_{ps}.png", img)


def get_contours_for_object():
    c = Config()
    e = Extractor()
    image = cv2.imread("_data/banana_grasping/image_0.png")
    e._extract_mask_rcnn(image, c.MASK_RCNN_THRESHOLD, "mask_rcnn_banana")
    e._extract_panoptic_fpn(image, c.MASK_RCNN_THRESHOLD, "panoptic_fpn_banana")
    e._extract_cascade_mask_rcnn(image, c.MASK_RCNN_THRESHOLD, "cascade_mask_rcnn_banana")

    image = cv2.imread("_data/hammer_grasping/image_0.png")
    e._extract_mask_rcnn(image, c.MASK_RCNN_THRESHOLD, "mask_rcnn_hammer")
    e._extract_panoptic_fpn(image, c.MASK_RCNN_THRESHOLD, "panoptic_fpn_hammer")
    e._extract_cascade_mask_rcnn(image, c.MASK_RCNN_THRESHOLD, "cascade_mask_rcnn_hammer")

    image = cv2.imread("_data/scissors_grasping/image_0.png")
    e._extract_mask_rcnn(image, c.MASK_RCNN_THRESHOLD, "mask_rcnn_scissors")
    e._extract_panoptic_fpn(image, c.MASK_RCNN_THRESHOLD, "panoptic_fpn_scissors")
    e._extract_cascade_mask_rcnn(image, c.MASK_RCNN_THRESHOLD, "cascade_mask_rcnn_scissors")


if __name__ == "__main__":
    pass
