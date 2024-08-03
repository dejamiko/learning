import json
import os

import numpy as np
from pytest import fixture, raises, mark

from config import Config
from playground.extractor import Extractor
from tm_utils import ImageEmbeddings


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
        json.dump([{config.IMAGE_EMBEDDINGS.value: expected}], f)
    embeddings = Extractor()(emb_dir, config)
    assert embeddings.shape == (1, 2, 3)
    assert np.allclose(embeddings[0], np.array(expected))


def test_extractor_some_embeddings_but_wrong_metric_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    # create embeddings of different metric
    other = [[1, 2, 3], [2, 4, 6]]
    with open(os.path.join(emb_dir, "embeddings.json"), "w") as f:
        json.dump([{"different metric": other}], f)
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
            [
                {config.IMAGE_EMBEDDINGS.value: expected[0]},
                {config.IMAGE_EMBEDDINGS.value: expected[1]},
                {config.IMAGE_EMBEDDINGS.value: expected[2]},
                {config.IMAGE_EMBEDDINGS.value: expected[3]},
                {config.IMAGE_EMBEDDINGS.value: expected[4]},
            ],
            f,
        )
    embeddings = Extractor()(emb_dir, config)
    assert embeddings.shape == (5, 2, 3)
    assert np.allclose(embeddings, np.array(expected))


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
        -0.35703462,
        1.19266963,
        -2.00560188,
        0.56439084,
        3.33149242,
        -0.28680599,
        0.3572948,
        -1.21180332,
    ]
    assert embeddings.shape == (5, 384)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_dino_2_full_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_2_FULL
    embeddings = Extractor()(emb_dir, config)
    expected = [
        2.19594669,
        2.36873817,
        1.9113816,
        0.00589865,
        -0.49594384,
        1.22531855,
        -0.76101184,
        1.61280763,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


@mark.skip(reason="The library cannot be installed without CUDA")
def test_extractor_no_embeddings_mamba_vision_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.MAMBA_VISION
    embeddings = Extractor()(emb_dir, config)
    expected = [
        2.19594669,
        2.36873817,
        1.9113816,
        0.00589865,
        -0.49594384,
        1.22531855,
        -0.76101184,
        1.61280763,
    ]
    assert embeddings.shape == (5, 640)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_vit_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.VIT
    embeddings = Extractor()(emb_dir, config)
    expected = [
        -0.09678458,
        0.06691732,
        -0.15958279,
        0.03637185,
        -0.03019533,
        0.37906396,
        0.01801646,
        -0.05673098,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_convnet_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.CONVNET
    embeddings = Extractor()(emb_dir, config)
    expected = [
        0.02492115,
        -0.41968155,
        0.10588291,
        0.42363903,
        0.16527481,
        0.35282275,
        -0.17214058,
        0.06407911,
    ]
    assert embeddings.shape == (5, 1024)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_no_embeddings_swin_works(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.SWIN
    embeddings = Extractor()(emb_dir, config)
    expected = [
        1.21000528,
        0.32423934,
        -0.27246797,
        0.8030616,
        -1.11264479,
        -0.62838131,
        -1.59494519,
        0.17526984,
    ]
    assert embeddings.shape == (5, 768)
    assert np.allclose(embeddings[0][2:10], expected)


def test_extractor_wrong_method_fails(empty_dir_and_config):
    emb_dir, config = empty_dir_and_config
    config.IMAGE_EMBEDDINGS = "different metric"
    with raises(ValueError) as e:
        Extractor()(emb_dir, config)
    assert str(e.value) == f"The method provided {config.IMAGE_EMBEDDINGS} is unknown."
