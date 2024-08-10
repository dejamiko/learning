import json
import os

import numpy as np
from pytest import fixture, raises

from config import Config
from playground.extractor import Extractor
from tm_utils import ImageEmbeddings, ContourImageEmbeddings, ImagePreprocessing


@fixture
def empty_dir_and_config():
    config = Config()
    config.LOAD_SIZE = 16
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_LAYER_9
    emb_dir = "tests/_test_assets/"
    emb_path = os.path.join(emb_dir, "embeddings.json")
    with open(emb_path) as f:
        existing_embeddings = json.load(f)
    os.remove(emb_path)
    yield emb_dir, config
    with open(emb_path, "w") as f:
        json.dump(existing_embeddings, f)


def test_extractor_no_embeddings_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.18434414,
        0.32338375,
        -0.81356609,
        -0.45523641,
        -0.32479423,
        0.41095853,
        1.04919481,
        0.11488045,
    ]
    assert embeddings.shape == (5, 6528)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_some_embeddings_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create embeddings for the same metric, but not all of them
    expected = [[1, 2, 3], [2, 4, 6]]
    with open(os.path.join(emb_dir, "embeddings.json"), "w") as f:
        json.dump(
            [{config.get_embedding_spec(): expected}],
            f,
        )
    embeddings = Extractor()(emb_dir, config)
    assert embeddings.shape == (1, 2, 3)
    assert np.allclose(embeddings[0], np.array(expected))


def test_extractor_some_embeddings_but_wrong_metric_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create embeddings of different metric
    other = [[1, 2, 3], [2, 4, 6]]
    with open(os.path.join(emb_dir, "embeddings.json"), "w") as f:
        json.dump([{f"different, []": other}], f)
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.18434414,
        0.32338375,
        -0.81356609,
        -0.45523641,
        -0.32479423,
        0.41095853,
        1.04919481,
        0.11488045,
    ]
    assert embeddings.shape == (5, 6528)
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
    with open(os.path.join(emb_dir, "embeddings.json"), "w") as f:
        json.dump(
            [{config.get_embedding_spec(): e} for e in expected],
            f,
        )
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
    with open(os.path.join(emb_dir, "embeddings.json"), "w") as f:
        json.dump(
            [{config.get_embedding_spec(): e} for e in expected],
            f,
        )
    embeddings = Extractor()(emb_dir, config)
    assert embeddings.shape == (5, 2, 3)
    assert np.allclose(embeddings, np.array(expected))


def test_extractor_all_embeddings_contour_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    # create all embeddings
    expected = [
        [[1, 2], [2, 4]],
        [[1.1, 2.1], [2.1, 4.1]],
        [[1.2, 2.2], [2.2, 4.2]],
        [[1.3, 2.3], [2.3, 4.3]],
        [[1.4, 2.4], [2.4, 4.4]],
    ]
    with open(os.path.join(emb_dir, "embeddings.json"), "w") as f:
        json.dump(
            [{config.get_embedding_spec(): e} for e in expected],
            f,
        )
    embeddings = Extractor()(emb_dir, config)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 5
    assert len(embeddings[0]) == 2
    assert len(embeddings[0][0]) == 2
    assert np.allclose(embeddings, expected)


def test_extractor_no_images_in_dir_fails(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create an empty directory and pass that in as img_dir
    empty_dir_path = os.path.join(emb_dir, "empty_dir/")
    with raises(ValueError) as e:
        Extractor()(empty_dir_path, config)
    assert str(e.value) == f"The directory provided {empty_dir_path} contains no images"


def test_extractor_no_embeddings_dino_layer_11_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_LAYER_11
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.79709041,
        -0.32174441,
        0.40859127,
        -0.62905365,
        -0.12319135,
        -0.09629215,
        0.14860496,
        0.10301444,
    ]
    assert embeddings.shape == (5, 6528)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dino_full_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_FULL
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -3.09272385,
        3.10472727,
        -1.42315328,
        0.87985611,
        0.58063781,
        -1.47418904,
        2.11999035,
        -2.34510541,
    ]
    assert embeddings.shape == (5, 384)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dino_2_full_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_2_FULL
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.96855688,
        1.35322404,
        1.6999315,
        0.06761098,
        -0.16643545,
        0.44783533,
        -0.03544366,
        1.30553341,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vit_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VIT
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.00308969,
        0.02598491,
        -0.03046953,
        -0.19576237,
        -0.26125684,
        0.18287084,
        -0.11704297,
        -0.0689105,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_convnet_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.CONVNET
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.06297413,
        -0.21367723,
        0.48447192,
        -0.44642058,
        0.2480932,
        0.46501365,
        -0.3112483,
        -0.204081,
    ]
    assert embeddings.shape == (5, 1024)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_swin_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.SWIN
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.65817058,
        -0.01108037,
        -0.17194957,
        0.44754627,
        -0.87932968,
        0.2345365,
        -1.51616478,
        0.10756487,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vit_msn_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VIT_MSN
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.19736178,
        0.67368895,
        0.42296565,
        -0.16271308,
        -0.08438256,
        -0.13777353,
        1.17792237,
        -0.44260886,
    ]
    assert embeddings.shape == (5, 384)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dobbe_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DOBBE
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.00000000e00,
        5.47229797e-02,
        0.00000000e00,
        4.96147841e-05,
        4.17437814e-02,
        7.57765546e-02,
        0.00000000e00,
        5.33931982e-03,
    ]
    assert embeddings.shape == (5, 512)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vc_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VC
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.03540432,
        0.05867103,
        0.06300978,
        -0.153437,
        0.02411774,
        -0.11084762,
        -0.02518206,
        -0.11365147,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


# def test_extractor_no_embeddings_own_models_works(empty_dir_and_config):
#     emb_dir, config = empty_dir_and_config
#     config.IMAGE_EMBEDDINGS = ImageEmbeddings.OWN_TRAINED
#     embeddings = Extractor()(emb_dir, config)
#     expected = [
#         -0.2170563,
#         -0.2170563,
#         -0.2170563,
#         -0.2170563,
#         -0.2170563,
#         -0.2170563,
#         -0.2170563,
#         -0.2170563,
#     ]
#     assert embeddings.shape == (5, 256**2 * 3)
#     assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_wrong_method_fails(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = "different metric"
    with raises(ValueError) as e:
        Extractor()(emb_dir, config)
    assert str(e.value) == f"The method provided {config.IMAGE_EMBEDDINGS} is unknown."


def test_all_embeddings_greyscale(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_PREPROCESSING = [ImagePreprocessing.GREYSCALE]
    for ie in ImageEmbeddings:
        config.IMAGE_EMBEDDINGS = ie
        Extractor()(emb_dir, config)
