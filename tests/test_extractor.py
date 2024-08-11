import json
import os

import cv2
import numpy as np
from pytest import fixture, raises

from config import Config
from playground.extractor import Extractor
from tm_utils import ImageEmbeddings, ContourImageEmbeddings, ImagePreprocessing


@fixture
def empty_dir_and_config():
    config = Config()
    config.LOAD_SIZE = 16
    emb_dir = "tests/_test_assets/"
    for d in os.listdir(emb_dir):
        if not d.endswith(".json"):
            continue
        os.remove(os.path.join(emb_dir, d))
    yield emb_dir, config
    for d in os.listdir(emb_dir):
        if not d.endswith(".json"):
            continue
        os.remove(os.path.join(emb_dir, d))


def test_extractor_some_embeddings_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create embeddings for the same metric, but not all of them
    expected = [[1, 2, 3], [2, 4, 6]]
    with open(
        os.path.join(emb_dir, f"embeddings_{config.get_embedding_spec()}.json"), "w"
    ) as f:
        json.dump(expected, f)
    embeddings = Extractor()(emb_dir, config)
    expected_new = np.array(
        [
            [3.10471344, 0.19727778, 1.66282141],
            [2.66162062, 0.22787094, 2.07419348],
            [2.78903913, -0.10260102, 2.68960094],
        ]
    )
    assert len(embeddings) == 5
    assert embeddings[0] == expected[0]
    assert embeddings[1] == expected[1]
    assert np.allclose(np.array(embeddings[2:])[:, 2:5], expected_new)


def test_extractor_some_embeddings_but_wrong_metric_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create embeddings of different metric
    other = [[1, 2, 3], [2, 4, 6]]
    with open(os.path.join(emb_dir, f"embeddings_different, [].json"), "w") as f:
        json.dump(other, f)
    embeddings = Extractor()(emb_dir, config)
    expected = [
        3.064394,
        -0.41076189,
        1.71749401,
        -1.14465475,
        1.17028403,
        -1.50916374,
        2.06556773,
        -0.6224308,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_all_embeddings_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create all embeddings
    expected = [
        [[1, 2, 3], [2, 4, 6]],
        [[1.1, 2.1, 3.1], [2.1, 4.1, 6.1]],
        [[1.2, 2.2, 3.2], [2.2, 4.2, 6.2]],
        [[1.3, 2.3, 3.3], [2.3, 4.3, 6.3]],
        [[1.4, 2.4, 3.4], [2.4, 4.4, 6.4]],
    ]
    with open(
        os.path.join(emb_dir, f"embeddings_{config.get_embedding_spec()}.json"), "w"
    ) as f:
        json.dump(expected, f)
    embeddings = Extractor()(emb_dir, config)
    assert embeddings.shape == (5, 2, 3)
    assert np.allclose(embeddings, np.array(expected))


def test_extractor_all_embeddings_different_preprocessing_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create all embeddings
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.GREYSCALE]
    expected = [
        [[1, 2, 3], [2, 4, 6]],
        [[1.1, 2.1, 3.1], [2.1, 4.1, 6.1]],
        [[1.2, 2.2, 3.2], [2.2, 4.2, 6.2]],
        [[1.3, 2.3, 3.3], [2.3, 4.3, 6.3]],
        [[1.4, 2.4, 3.4], [2.4, 4.4, 6.4]],
    ]
    with open(
        os.path.join(emb_dir, f"embeddings_{config.get_embedding_spec()}.json"), "w"
    ) as f:
        json.dump(expected, f)
    embeddings = Extractor()(emb_dir, config)
    assert embeddings.shape == (5, 2, 3)
    assert np.allclose(embeddings, np.array(expected))


def test_extractor_all_embeddings_contour_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    # create all embeddings
    expected = [
        [[1, 2], [2, 4], [3, 6]],
        [[1.1, 2.1], [2.1, 4.1]],
        [[1.2, 2.2], [2.2, 4.2]],
        [[1.3, 2.3], [2.3, 4.3]],
        [[1.4, 2.4], [2.4, 4.4]],
    ]
    with open(
        os.path.join(emb_dir, f"embeddings_{config.get_embedding_spec()}.json"), "w"
    ) as f:
        json.dump(expected, f)
    embeddings = Extractor()(emb_dir, config)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 5
    assert len(embeddings[0]) == 3
    assert len(embeddings[0][0]) == 2
    for i in range(5):
        assert np.allclose(embeddings[i], expected[i])


def test_extractor_no_images_in_dir_fails(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create an empty directory and pass that in as img_dir
    empty_dir_path = os.path.join(emb_dir, "empty_dir/")
    with raises(ValueError) as e:
        Extractor()(empty_dir_path, config)
    assert str(e.value) == f"The directory provided {empty_dir_path} contains no images"


def test_extractor_no_embeddings_dino_layer_9_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_LAYER_9
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.21889894,
        0.37965849,
        0.11891212,
        -0.20318767,
        -0.34644514,
        1.01547039,
        1.21056187,
        -0.7112096,
    ]
    assert embeddings.shape == (5, 6528)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dino_layer_11_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_LAYER_11
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.79233724,
        -0.6284228,
        -0.05257636,
        -0.40432939,
        -0.33743536,
        -0.06903843,
        0.23262756,
        0.59205562,
    ]
    assert embeddings.shape == (5, 6528)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dino_full_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_FULL
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.71678388,
        1.62074006,
        0.09509128,
        0.21747777,
        3.27846622,
        -1.0529964,
        1.07123089,
        -2.03937387,
    ]
    assert embeddings.shape == (5, 384)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dino_2_full_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_2_FULL
    embeddings = Extractor()(emb_dir, config)
    expected = [
        3.064394,
        -0.41076189,
        1.71749401,
        -1.14465475,
        1.17028403,
        -1.50916374,
        2.06556773,
        -0.6224308,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vit_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VIT
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.06745528,
        -0.12765129,
        0.09235362,
        -0.18822737,
        -0.27684909,
        0.11348941,
        -0.1894778,
        0.03964938,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_convnet_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.CONVNET
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.16322817,
        -0.25928095,
        0.3310194,
        -0.36358601,
        0.30661699,
        0.46384785,
        -0.29757681,
        -0.11246782,
    ]
    assert embeddings.shape == (5, 1024)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_swin_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.SWIN
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.95772016,
        0.23786928,
        0.25221035,
        0.52895129,
        -1.00214303,
        0.11961905,
        -1.30438244,
        -0.05033324,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vit_msn_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VIT_MSN
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.10130542,
        0.63633168,
        0.49498594,
        -0.13132143,
        -0.00280582,
        -0.14108324,
        1.15758228,
        -0.5035302,
    ]
    assert embeddings.shape == (5, 384)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dobbe_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DOBBE
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.0,
        0.11009531,
        0.0,
        0.0,
        0.04350008,
        0.12214363,
        0.00494772,
        0.01960813,
    ]
    assert embeddings.shape == (5, 512)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vc_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VC
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.0265067,
        0.07708694,
        0.07868653,
        -0.16145553,
        0.04594788,
        -0.0877109,
        -0.05015997,
        -0.10860752,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_own_models_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.OWN_TRAINED
    embeddings = Extractor()(emb_dir, config)
    assert len(embeddings) == 5
    for i in range(len(embeddings)):
        assert len(embeddings[i]) == 0


def test_extractor_wrong_method_fails(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImagePreprocessing.GREYSCALE
    with raises(ValueError) as e:
        Extractor()(emb_dir, config)
    assert str(e.value) == f"The method provided {config.IMAGE_EMBEDDINGS} is unknown."


def test_preprocessing_nothing_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = []
    image = cv2.imread(os.path.join(emb_dir, "image_0.png"))
    new_image = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )
    assert image.shape == new_image.shape
    assert np.allclose(image, new_image)


def test_preprocessing_background_removal_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.BACKGROUND_REM]
    image = cv2.imread(os.path.join(emb_dir, "image_0.png"))
    new_image = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )
    assert image.shape == new_image.shape
    unique_colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    most_common_color_image = unique_colors[np.argmax(counts)]

    unique_colors, counts = np.unique(
        new_image.reshape(-1, 3), axis=0, return_counts=True
    )
    most_common_color_new_image = unique_colors[np.argmax(counts)]

    assert np.allclose(
        most_common_color_new_image, (0, 0, 0)
    )  # the background should be black
    assert most_common_color_image not in unique_colors


def test_preprocessing_segmentation_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.SEGMENTATION]
    image = cv2.imread(os.path.join(emb_dir, "image_0.png"))
    new_image = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )
    assert image.shape == new_image.shape
    unique_colors_old, counts_old = np.unique(
        image.reshape(-1, 3), axis=0, return_counts=True
    )
    most_common_color_image = unique_colors_old[np.argmax(counts_old)]

    unique_colors, counts = np.unique(
        new_image.reshape(-1, 3), axis=0, return_counts=True
    )
    most_common_color_new_image = unique_colors[np.argmax(counts)]

    assert np.allclose(
        most_common_color_new_image, (0, 0, 0)
    )  # the background should be black
    index = np.where(unique_colors == most_common_color_image)[0][0]
    assert counts[index] * 10 < counts_old.max()


def test_preprocessing_greyscale_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.GREYSCALE]
    image = cv2.imread(os.path.join(emb_dir, "image_0.png"))
    new_image = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )
    assert image.shape == new_image.shape
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            assert len(set(new_image[i][j])) == 1


def test_preprocessing_cropping_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.CROPPING]
    for i in range(5):
        image = cv2.imread(os.path.join(emb_dir, f"image_{i}.png"))
        new_image = Extractor._apply_preprocessing(
            config, image, f"_data/000_hammering/image_{i}.png"
        )
        assert image.shape != new_image.shape
        assert image.shape[0] * 0.8 == new_image.shape[0]
        assert image.shape[1] * 0.8 == new_image.shape[1]
        assert image.shape[2] == new_image.shape[2]


def test_preprocessing_cropping_pushing_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.CROPPING]
    for i in range(5):
        image = cv2.imread(os.path.join(emb_dir, f"image_{i}.png"))
        new_image = Extractor._apply_preprocessing(
            config, image, f"_data/033_pushing/image_{i}.png"
        )
        assert image.shape != new_image.shape
        assert image.shape[0] * 0.8 == new_image.shape[0]
        assert image.shape[1] * 0.8 == new_image.shape[1]
        assert image.shape[2] == new_image.shape[2]


def test_preprocessing_cropping_absolute_path_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.CROPPING]
    for i in range(5):
        image = cv2.imread(os.path.join(emb_dir, f"image_{i}.png"))
        new_image = Extractor._apply_preprocessing(
            config,
            image,
            f"/Users/mikolajdeja/Coding/learning/_data/011_hammering/image_{i}.png",
        )
        assert image.shape != new_image.shape
        assert image.shape[0] * 0.8 == new_image.shape[0]
        assert image.shape[1] * 0.8 == new_image.shape[1]
        assert image.shape[2] == new_image.shape[2]


def test_preprocessing_list_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [
        ImagePreprocessing.CROPPING,
        ImagePreprocessing.GREYSCALE,
    ]
    image = cv2.imread(os.path.join(emb_dir, "image_0.png"))
    new_image = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )
    assert image.shape != new_image.shape
    assert image.shape[0] * 0.8 == new_image.shape[0]
    assert image.shape[1] * 0.8 == new_image.shape[1]
    assert image.shape[2] == new_image.shape[2]

    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            assert len(set(new_image[i][j])) == 1


def test_preprocessing_cropping_greyscale_both_orders_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [
        ImagePreprocessing.CROPPING,
        ImagePreprocessing.GREYSCALE,
    ]
    image = cv2.imread(os.path.join(emb_dir, "image_0.png"))
    new_image = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )

    config.IMAGE_PREPROCESSING = [
        ImagePreprocessing.GREYSCALE,
        ImagePreprocessing.CROPPING,
    ]
    new_image_2 = Extractor._apply_preprocessing(
        config, image, "_data/000_hammering/image_0.png"
    )

    assert np.allclose(new_image, new_image_2)
