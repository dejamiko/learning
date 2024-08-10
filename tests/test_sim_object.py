import numpy as np
from pytest import fixture

from config import Config
from playground.sim_object import SimObject
from tm_utils import Task


def test_object_init_works():
    config = Config()
    _ = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets",
    )


@fixture
def object_fixture():
    config = Config()
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets",
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "tests/_test_assets",
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
    assert obj_0.image_path == "tests/_test_assets"
    assert obj_1.image_path == "tests/_test_assets"


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

    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.7400139849342894)
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

    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.6954608364472726)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)
