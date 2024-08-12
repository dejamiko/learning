import os

import numpy as np
from pytest import fixture

from config import Config
from playground.sim_object import SimObject
from tm_utils import Task, NNSimilarityMeasure, ImageEmbeddings


def test_object_init_works():
    config = Config()
    _ = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets/obj_1_img",
    )


@fixture
def object_fixture():
    config = Config()
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets/obj_1_img",
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "tests/_test_assets/obj_2_img",
    )
    obj_0.visible_repr = [0.1, 0.2, 0.3]
    obj_1.visible_repr = [1.1, 1.2, 1.3]
    return obj_0, obj_1, config


def test_object_fields_work(object_fixture):
    obj_0, obj_1, c = object_fixture

    assert obj_0.index == 0
    assert obj_1.index == 1
    assert obj_0.name == "object_0"
    assert obj_1.name == "object_1"
    assert obj_0.task == Task.HAMMERING
    assert obj_1.task == Task.HAMMERING
    assert obj_0.c == c
    assert obj_1.c == c
    assert np.allclose(obj_0.visible_repr, [0.1, 0.2, 0.3])
    assert np.allclose(obj_1.visible_repr, [1.1, 1.2, 1.3])
    assert obj_0.image_path == "tests/_test_assets/obj_1_img"
    assert obj_1.image_path == "tests/_test_assets/obj_2_img"


def test_get_visual_similarity_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    obj_0.visible_repr = np.zeros_like(obj_0.visible_repr)
    obj_0.visible_repr[-1] = 1
    obj_1.visible_repr = np.zeros_like(obj_1.visible_repr)
    obj_1.visible_repr[0] = 1
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.0)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_str_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert str(obj_0) == "object_0, [0.1, 0.2, 0.3], hammering"

    assert str(obj_1) == "object_1, [1.1, 1.2, 1.3], hammering"


def test_repr_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert repr(obj_0) == "object_0, [0.1, 0.2, 0.3], hammering"

    assert repr(obj_1) == "object_1, [1.1, 1.2, 1.3], hammering"


def test_actual_embeddings_sim_works():
    config = Config()
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets/obj_1_img",
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "tests/_test_assets/obj_2_img",
    )

    assert obj_0.visible_repr.shape == (768,)
    assert obj_1.visible_repr.shape == (768,)

    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.706240539207901)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_actual_embeddings_all_images_sim_works():
    config = Config()
    config.USE_ALL_IMAGES = True
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets/obj_1_img",
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "tests/_test_assets/obj_2_img",
    )

    assert obj_0.visible_repr.shape == (5, 768)
    assert obj_1.visible_repr.shape == (5, 768)

    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.7295830412191917)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


@fixture
def siamese_obj_fixture():
    config = Config()
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.SIAMESE
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "_data/000_hammering/",
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "_data/090_hammering/",
    )
    yield obj_0, obj_1, config
    os.remove(os.path.join("_data/000_hammering", "embeddings_siamese, [].json"))
    os.remove(os.path.join("_data/090_hammering", "embeddings_siamese, [].json"))


@fixture
def siamese_obj_fixture_all():
    config = Config()
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.SIAMESE
    config.USE_ALL_IMAGES = True
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "_data/000_hammering/",
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "_data/090_hammering/",
    )
    yield obj_0, obj_1, config
    os.remove(os.path.join("_data/000_hammering", "embeddings_siamese, [].json"))
    os.remove(os.path.join("_data/090_hammering", "embeddings_siamese, [].json"))


def test_actual_siamese_trained_similarity_works(siamese_obj_fixture):
    obj_0, obj_1, config = siamese_obj_fixture
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.TRAINED

    assert len(obj_0.visible_repr) == 3
    assert len(obj_1.visible_repr) == 3

    assert obj_0.get_visual_similarity(obj_1) == 0.4545803666114807
    assert obj_0.get_visual_similarity(obj_0) == 0.9999999403953552
    assert obj_1.get_visual_similarity(obj_1) == 1


def test_actual_siamese_fine_tuned_similarity_works(siamese_obj_fixture):
    obj_0, obj_1, config = siamese_obj_fixture
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.FINE_TUNED

    assert len(obj_0.visible_repr) == 3
    assert len(obj_1.visible_repr) == 3

    assert obj_0.get_visual_similarity(obj_1) == 0.9963110685348511
    assert obj_0.get_visual_similarity(obj_0) == 1
    assert obj_1.get_visual_similarity(obj_1) == 1


def test_actual_siamese_linearly_probed_similarity_works(siamese_obj_fixture):
    obj_0, obj_1, config = siamese_obj_fixture
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.LINEARLY_PROBED

    assert len(obj_0.visible_repr) == 3
    assert len(obj_1.visible_repr) == 3

    assert obj_0.get_visual_similarity(obj_1) == 0.87115079164505
    assert obj_0.get_visual_similarity(obj_0) == 1
    assert obj_1.get_visual_similarity(obj_1) == 1


def test_actual_siamese_trained_similarity_all_works(siamese_obj_fixture_all):
    obj_0, obj_1, config = siamese_obj_fixture_all
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.TRAINED

    assert len(obj_0.visible_repr) == 5
    assert len(obj_1.visible_repr) == 5

    assert obj_0.get_visual_similarity(obj_1) == 0.4545803666114807
    assert obj_0.get_visual_similarity(obj_0) == 0.9999999403953552
    assert obj_1.get_visual_similarity(obj_1) == 1


def test_actual_siamese_fine_tuned_similarity_all_works(siamese_obj_fixture_all):
    obj_0, obj_1, config = siamese_obj_fixture_all
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.FINE_TUNED

    assert len(obj_0.visible_repr) == 5
    assert len(obj_1.visible_repr) == 5

    assert obj_0.get_visual_similarity(obj_1) == 0.9963110685348511
    assert obj_0.get_visual_similarity(obj_0) == 1
    assert obj_1.get_visual_similarity(obj_1) == 1


def test_actual_siamese_linearly_probed_similarity_all_works(siamese_obj_fixture_all):
    obj_0, obj_1, config = siamese_obj_fixture_all
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.LINEARLY_PROBED

    assert len(obj_0.visible_repr) == 5
    assert len(obj_1.visible_repr) == 5

    assert obj_0.get_visual_similarity(obj_1) == 0.87115079164505
    assert obj_0.get_visual_similarity(obj_0) == 1
    assert obj_1.get_visual_similarity(obj_1) == 1
