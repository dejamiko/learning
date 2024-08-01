import numpy as np
from pytest import fixture, raises

from config import Config
from playground.basic_object import BasicObject
from utils import Task, get_rng


def test_object_init_works():
    config = Config()
    _ = BasicObject(
        0, config, Task.HAMMERING, np.zeros(config.LATENT_DIM), get_rng(config.SEED)
    )


def test_object_init_with_incorrect_task_fails():
    config = Config()
    invalid_task = "invalid_task"
    with raises(ValueError) as e:
        BasicObject(
            0, config, invalid_task, np.zeros(config.LATENT_DIM), get_rng(config.SEED)
        )
    assert str(e.value) == f"'{invalid_task}' is not a valid Task"


def test_object_init_with_incorrect_latent_representation_type_fails():
    config = Config()
    with raises(AssertionError) as e:
        BasicObject(
            0, config, Task.HAMMERING, [0] * config.LATENT_DIM, get_rng(config.SEED)
        )
    assert str(e.value) == f"Expected np.ndarray, got {list}"


def test_object_init_with_incorrect_latent_representation_dimensionality_fails():
    config = Config()
    incorrect_shape = (config.LATENT_DIM, config.LATENT_DIM)
    with raises(AssertionError) as e:
        BasicObject(
            0, config, Task.HAMMERING, np.zeros(incorrect_shape), get_rng(config.SEED)
        )
    assert str(e.value) == f"Expected 1D array, got {len(incorrect_shape)}D"


def test_object_init_with_incorrect_latent_representation_shape_fails():
    config = Config()
    incorrect_shape = config.LATENT_DIM - 1
    with raises(AssertionError) as e:
        BasicObject(
            0, config, Task.HAMMERING, np.zeros(incorrect_shape), get_rng(config.SEED)
        )
    assert (
        str(e.value)
        == f"Expected array of length {config.LATENT_DIM}, got {incorrect_shape}"
    )


@fixture
def object_fixture():
    config = Config()
    np.random.seed(config.SEED)
    obj_0 = BasicObject(
        0, config, Task.HAMMERING, np.zeros(config.LATENT_DIM), get_rng(config.SEED)
    )
    np.random.seed(config.SEED)
    obj_1 = BasicObject(
        1, config, Task.HAMMERING, np.ones(config.LATENT_DIM), get_rng(config.SEED)
    )
    return obj_0, obj_1, config


def test_object_fields_work(object_fixture):
    obj_0, obj_1, c = object_fixture

    assert obj_0.index == 0
    assert obj_1.index == 1
    assert obj_0.name == "Object 0"
    assert obj_1.name == "Object 1"
    assert obj_0.task == Task.HAMMERING
    assert obj_1.task == Task.HAMMERING
    assert obj_0.c == c
    assert obj_1.c == c
    assert np.allclose(obj_0.latent_repr, np.zeros(c.LATENT_DIM))
    assert np.allclose(obj_1.latent_repr, np.ones(c.LATENT_DIM))

    assert np.allclose(
        obj_0.visible_repr,
        [
            0.01257302,
            -0.01321049,
            0.06404227,
            0.01049001,
            -0.05356694,
            0.03615951,
            0.1304,
            0.0947081,
            -0.07037352,
            -0.12654215,
        ],
    )
    assert np.allclose(
        obj_1.visible_repr,
        [
            1.01257302,
            0.98678951,
            1.06404227,
            1.01049001,
            0.94643306,
            1.03615951,
            1.1304,
            1.0947081,
            0.92962648,
            0.87345785,
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
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.18602748506878572)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_str_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert (
        str(obj_0) == "Object 0 ([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]), "
        "[ 0.01257302 -0.01321049  0.06404227  0.01049001 -0.05356694  0.03615951\n  "
        "0.1304      0.0947081  -0.07037352 -0.12654215], hammering"
    )

    assert (
        str(obj_1) == "Object 1 ([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]), "
        "[1.01257302 0.98678951 1.06404227 1.01049001 0.94643306 1.03615951\n "
        "1.1304     1.0947081  0.92962648 0.87345785], hammering"
    )


def test_repr_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    assert (
        repr(obj_0) == "Object 0 ([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]), "
        "[ 0.01257302 -0.01321049  0.06404227  0.01049001 -0.05356694  0.03615951\n  "
        "0.1304      0.0947081  -0.07037352 -0.12654215], hammering"
    )

    assert (
        repr(obj_1) == "Object 1 ([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]), "
        "[1.01257302 0.98678951 1.06404227 1.01049001 0.94643306 1.03615951\n "
        "1.1304     1.0947081  0.92962648 0.87345785], hammering"
    )
