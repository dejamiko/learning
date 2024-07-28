import numpy as np
from pytest import fixture

from config import Config
from playground.storage import ObjectStorage
from playground.task_types import Task


def test_init_storage_works():
    config = Config()
    _ = ObjectStorage(config)


@fixture
def empty_storage_fixture():
    config = Config()
    storage = ObjectStorage(config)
    storage.reset(config)
    return storage, config


def test_storage_fields_work(empty_storage_fixture):
    storage, config = empty_storage_fixture

    assert storage.c == config
    assert storage._objects is None
    assert storage._visual_similarities_by_task is None
    assert storage._latent_similarities is None


def test_generate_random_objects_works(empty_storage_fixture):
    storage, c = empty_storage_fixture
    c.USE_REAL_OBJECTS = False
    storage.generate_objects()

    assert storage._objects is not None
    assert storage._objects.shape == (c.OBJ_NUM,)
    assert storage._visual_similarities_by_task is not None
    assert len(storage._visual_similarities_by_task) == 3
    assert storage._visual_similarities_by_task[Task.GRASPING.value].shape == (
        c.OBJ_NUM,
        c.OBJ_NUM,
    )
    assert storage._latent_similarities is not None
    assert storage._latent_similarities.shape == (c.OBJ_NUM, c.OBJ_NUM)


@fixture
def storage_fixture_random():
    config = Config()
    config.SEED = 1
    config.USE_REAL_OBJECTS = False
    storage = ObjectStorage(config)
    storage.reset(config)
    storage.generate_objects()
    return storage, config


def test_get_visual_similarity_works(storage_fixture_random):
    storage, c = storage_fixture_random
    assert np.allclose(storage.get_visual_similarity(0, 1), 0)
    assert np.allclose(storage.get_visual_similarity(0, 2), 0.5041701285557193)


def test_get_true_success_probability_boolean_works(storage_fixture_random):
    storage, c = storage_fixture_random
    c.SUCCESS_RATE_BOOLEAN = True
    assert storage.get_true_success_probability(0, 1, 0.4) == 0.0
    assert storage.get_true_success_probability(0, 2, 0.45) == 1.0
    assert storage.get_true_success_probability(0, 2, 0.5) == 0.0


def test_get_true_success_probability_real_works(storage_fixture_random):
    storage, c = storage_fixture_random
    c.SUCCESS_RATE_BOOLEAN = False
    assert storage.get_true_success_probability(0, 1, 0.4) == 0.0
    assert storage.get_true_success_probability(0, 2, 0.4) == 0.48498082292769984
    assert storage.get_true_success_probability(0, 2, 0.7) == 0.48498082292769984


def test_estimated_success_probability_boolean_works(storage_fixture_random):
    storage, c = storage_fixture_random
    c.SUCCESS_RATE_BOOLEAN = True
    assert storage.get_estimated_success_probability(0, 1, 0.4) == 0.0
    assert storage.get_estimated_success_probability(0, 2, 0.5) == 1.0
    assert storage.get_estimated_success_probability(0, 2, 0.55) == 0.0


def test_estimated_success_probability_real_works(storage_fixture_random):
    storage, c = storage_fixture_random
    c.SUCCESS_RATE_BOOLEAN = False
    assert storage.get_estimated_success_probability(0, 16, 0.4) == 0.0
    assert storage.get_estimated_success_probability(0, 2, 0.15) == 0.5041701285557193
    assert storage.get_estimated_success_probability(0, 2, 0.7) == 0.5041701285557193


def test_get_objects_works(storage_fixture_random):
    storage, c = storage_fixture_random
    assert storage.get_objects() is not None
    assert len(storage.get_objects()) == c.OBJ_NUM
    assert all(storage.get_objects() == storage._objects)


def test_ingest_real_objects_works(empty_storage_fixture):
    storage, c = empty_storage_fixture
    c.USE_REAL_OBJECTS = True
    storage.generate_objects()

    assert storage._objects is not None
    assert storage._objects.shape == (c.OBJ_NUM,)
    assert storage._visual_similarities_by_task is not None
    assert len(storage._visual_similarities_by_task) == 3
    assert storage._visual_similarities_by_task[Task.GRASPING.value].shape == (
        c.OBJ_NUM,
        c.OBJ_NUM,
    )
    assert storage._latent_similarities is not None
    assert storage._latent_similarities.shape == (c.OBJ_NUM, c.OBJ_NUM)


@fixture
def storage_fixture_real():
    config = Config()
    config.SEED = 1
    config.USE_REAL_OBJECTS = True
    storage = ObjectStorage(config)
    storage.reset(config)
    storage.generate_objects()
    return storage, config


def test_get_visual_similarity_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    assert np.allclose(storage.get_visual_similarity(0, 16), 0)
    assert np.allclose(storage.get_visual_similarity(0, 3), 0.15456099522042774)


def test_get_true_success_probability_boolean_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    c.SUCCESS_RATE_BOOLEAN = True
    assert storage.get_true_success_probability(0, 16, 0.4) == 0.0
    assert storage.get_true_success_probability(0, 3, 0.59) == 1.0
    assert storage.get_true_success_probability(0, 3, 0.61) == 0.0


def test_get_true_success_probability_real_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    c.SUCCESS_RATE_BOOLEAN = False
    assert storage.get_true_success_probability(0, 16, 0.4) == 0.0
    assert storage.get_true_success_probability(0, 3, 0.4) == 0.6
    assert storage.get_true_success_probability(0, 3, 0.7) == 0.6


def test_estimated_success_probability_boolean_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    c.SUCCESS_RATE_BOOLEAN = True
    assert storage.get_estimated_success_probability(0, 16, 0.4) == 0.0
    assert storage.get_estimated_success_probability(0, 3, 0.15) == 1.0
    assert storage.get_estimated_success_probability(0, 3, 0.2) == 0.0


def test_estimated_success_probability_real_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    c.SUCCESS_RATE_BOOLEAN = False
    assert storage.get_estimated_success_probability(0, 16, 0.4) == 0.0
    assert storage.get_estimated_success_probability(0, 3, 0.15) == 0.15456099522042774
    assert storage.get_estimated_success_probability(0, 3, 0.2) == 0.15456099522042774


def test_get_objects_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    assert storage.get_objects() is not None
    assert len(storage.get_objects()) == c.OBJ_NUM
    assert all(storage.get_objects() == storage._objects)
