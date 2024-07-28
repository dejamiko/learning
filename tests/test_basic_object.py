import numpy as np
from pytest import fixture, raises

from config import Config
from playground.basic_object import BasicObject
from playground.task_types import Task


def test_object_init_works():
    config = Config()
    _ = BasicObject(0, config, Task.HAMMERING.value, np.zeros(config.LATENT_DIM))


def test_object_init_with_incorrect_task_fails():
    config = Config()
    invalid_task = "invalid_task"
    with raises(AssertionError) as e:
        BasicObject(0, config, invalid_task, np.zeros(config.LATENT_DIM))
    assert (
        str(e.value) == f"The task provided {invalid_task} is not a valid Task {Task}"
    )


def test_object_init_with_incorrect_latent_representation_type_fails():
    config = Config()
    with raises(AssertionError) as e:
        BasicObject(0, config, Task.HAMMERING.value, [0] * config.LATENT_DIM)
    assert str(e.value) == f"Expected np.ndarray, got {list}"


def test_object_init_with_incorrect_latent_representation_dimensionality_fails():
    config = Config()
    incorrect_shape = (config.LATENT_DIM, config.LATENT_DIM)
    with raises(AssertionError) as e:
        BasicObject(0, config, Task.HAMMERING.value, np.zeros(incorrect_shape))
    assert str(e.value) == f"Expected 1D array, got {len(incorrect_shape)}D"


def test_object_init_with_incorrect_latent_representation_shape_fails():
    config = Config()
    incorrect_shape = config.LATENT_DIM - 1
    with raises(AssertionError) as e:
        BasicObject(0, config, Task.HAMMERING.value, np.zeros(incorrect_shape))
    assert (
        str(e.value)
        == f"Expected array of length {config.LATENT_DIM}, got {incorrect_shape}"
    )


@fixture
def object_fixture():
    config = Config()
    np.random.seed(config.SEED)
    obj_0 = BasicObject(0, config, Task.HAMMERING.value, np.zeros(config.LATENT_DIM))
    np.random.seed(config.SEED)
    obj_1 = BasicObject(1, config, Task.HAMMERING.value, np.ones(config.LATENT_DIM))
    return obj_0, obj_1, config


def test_object_fields_work(object_fixture):
    obj_0, obj_1, c = object_fixture

    assert obj_0.index == 0
    assert obj_1.index == 1
    assert obj_0.name == "Object 0"
    assert obj_1.name == "Object 1"
    assert obj_0.task == Task.HAMMERING.value
    assert obj_1.task == Task.HAMMERING.value
    assert obj_0.c == c
    assert obj_1.c == c
    assert np.allclose(obj_0.latent_repr, np.zeros(c.LATENT_DIM))
    assert np.allclose(obj_1.latent_repr, np.ones(c.LATENT_DIM))

    assert np.allclose(
        obj_0.visible_repr,
        [
            0.17640523,
            0.04001572,
            0.0978738,
            0.22408932,
            0.1867558,
            -0.09772779,
            0.09500884,
            -0.01513572,
            -0.01032189,
            0.04105985,
        ],
    )
    assert np.allclose(
        obj_1.visible_repr,
        [
            1.17640523,
            1.04001572,
            1.0978738,
            1.22408932,
            1.1867558,
            0.90227221,
            1.09500884,
            0.98486428,
            0.98967811,
            1.04105985,
        ],
    )


def test_get_latent_similarity_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    obj_0.latent_repr = np.zeros_like(obj_0.latent_repr)
    obj_0.latent_repr[-1] = 1
    obj_1.latent_repr = np.zeros_like(obj_1.latent_repr)
    obj_1.latent_repr[0] = 1
    assert np.allclose(obj_0.get_latent_similarity(obj_1), 0)
    assert np.allclose(obj_0.get_latent_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_latent_similarity(obj_1), 1)


def test_get_visual_similarity_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.6755281920385012)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_str_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert (
        str(obj_0) == "Object 0 ([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]), "
        "[ 0.17640523  0.04001572  0.0978738   0.22408932  0.1867558  -0.09772779\n  "
        "0.09500884 -0.01513572 -0.01032189  0.04105985], hammering"
    )

    assert (
        str(obj_1) == "Object 1 ([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]), "
        "[1.17640523 1.04001572 1.0978738  1.22408932 1.1867558  0.90227221\n "
        "1.09500884 0.98486428 0.98967811 1.04105985], hammering"
    )


def test_repr_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert (
        repr(obj_0) == "Object 0 ([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]), "
        "[ 0.17640523  0.04001572  0.0978738   0.22408932  0.1867558  -0.09772779\n  "
        "0.09500884 -0.01513572 -0.01032189  0.04105985], hammering"
    )

    assert (
        repr(obj_1) == "Object 1 ([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]), "
        "[1.17640523 1.04001572 1.0978738  1.22408932 1.1867558  0.90227221\n "
        "1.09500884 0.98486428 0.98967811 1.04105985], hammering"
    )
