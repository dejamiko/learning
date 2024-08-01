import os

import cv2
import numpy as np
from pytest import fixture

from config import Config
from playground.sim_object import SimObject
from utils import Task


def test_object_init_works():
    config = Config()
    _ = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets",
        [np.zeros((1, 4))],
    )


@fixture
def object_fixture():
    config = Config()
    sp = (1, 4)
    obj_0 = SimObject(
        0,
        config,
        Task.HAMMERING,
        "object_0",
        "tests/_test_assets",
        [np.zeros(sp), np.zeros(sp), np.zeros(sp)],
    )
    obj_1 = SimObject(
        1,
        config,
        Task.HAMMERING,
        "object_1",
        "tests/_test_assets",
        [np.ones(sp), np.ones(sp), np.ones(sp)],
    )
    return obj_0, obj_1, config


@fixture
def images_fixture():
    images = []
    img_path = "tests/_test_assets"
    ims = ["image_0.png", "image_1.png", "image_2.png", "image_3.png", "image_4.png"]
    for im in ims:
        images.append(cv2.imread(os.path.join(img_path, im)))
    return images


def test_object_fields_work(object_fixture, images_fixture):
    obj_0, obj_1, c = object_fixture

    sp = (1, 4)

    assert obj_0.index == 0
    assert obj_1.index == 1
    assert obj_0.name == "object_0"
    assert obj_1.name == "object_1"
    assert obj_0.task == Task.HAMMERING
    assert obj_1.task == Task.HAMMERING
    assert obj_0.c == c
    assert obj_1.c == c
    assert np.allclose(
        obj_0.image_embeddings, [np.zeros(sp), np.zeros(sp), np.zeros(sp)]
    )
    assert np.allclose(obj_1.image_embeddings, [np.ones(sp), np.ones(sp), np.ones(sp)])
    assert np.allclose(obj_0.images, images_fixture)


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
    obj_0, obj_1, c = object_fixture  # a
    assert str(obj_0) == "object_0, [0. 0. 0. 0.], hammering"

    assert str(obj_1) == "object_1, [1. 1. 1. 1.], hammering"


def test_repr_works(object_fixture):
    obj_0, obj_1, c = object_fixture  # a
    assert repr(obj_0) == "object_0, [0. 0. 0. 0.], hammering"

    assert repr(obj_1) == "object_1, [1. 1. 1. 1.], hammering"
