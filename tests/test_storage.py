import numpy as np
from pytest import fixture

from config import Config
from playground.storage import ObjectStorage


def test_init_storage_works():
    config = Config()
    _ = ObjectStorage(config)


@fixture
def empty_storage_fixture():
    config = Config()
    config.OBJ_NUM = 100
    storage = ObjectStorage(config)
    return storage, config


def test_storage_fields_work(empty_storage_fixture):
    storage, config = empty_storage_fixture

    assert storage.c == config
    assert storage._objects is None
    assert storage._visual_similarities is None
    assert storage._latent_similarities is None


def test_generate_random_objects_works(empty_storage_fixture):
    storage, c = empty_storage_fixture
    c.USE_REAL_OBJECTS = False
    storage.generate_objects()

    assert storage._objects is not None
    assert storage._objects.shape == (c.OBJ_NUM,)
    assert storage._visual_similarities is not None
    assert storage._visual_similarities.shape == (
        c.OBJ_NUM,
        c.OBJ_NUM,
    )
    assert storage._latent_similarities is not None
    assert storage._latent_similarities.shape == (c.OBJ_NUM, c.OBJ_NUM)


@fixture
def storage_fixture_random():
    config = Config()
    config.OBJ_NUM = 100
    config.USE_REAL_OBJECTS = False
    storage = ObjectStorage(config)
    storage.generate_objects()
    return storage, config


def test_get_visual_similarity_works(storage_fixture_random):
    storage, c = storage_fixture_random
    assert np.allclose(storage.get_visual_similarity(0, 1), 0)
    assert np.allclose(storage.get_visual_similarity(0, 2), 0.6297178434321439)


def test_get_true_success_probability_boolean_works(storage_fixture_random):
    storage, c = storage_fixture_random
    c.SUCCESS_RATE_BOOLEAN = True
    assert storage.get_true_success_probability(0, 1, 0.4) == 0.0
    assert storage.get_true_success_probability(0, 2, 0.6) == 1.0
    assert storage.get_true_success_probability(0, 2, 0.65) == 0.0


def test_get_true_success_probability_real_works(storage_fixture_random):
    storage, c = storage_fixture_random
    c.SUCCESS_RATE_BOOLEAN = False
    assert storage.get_true_success_probability(0, 1, 0.4) == 0.0
    assert storage.get_true_success_probability(0, 2, 0.4) == 0.6162264663089067
    assert storage.get_true_success_probability(0, 2, 0.7) == 0.6162264663089067


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
    assert storage._objects.shape == (51,)
    assert storage._visual_similarities is not None
    assert storage._visual_similarities.shape == (
        len(storage._objects),
        len(storage._objects),
    )
    assert storage._latent_similarities is not None
    assert storage._latent_similarities.shape == (
        len(storage._objects),
        len(storage._objects),
    )


@fixture
def storage_fixture_real():
    config = Config()
    config.SEED = 1
    config.USE_REAL_OBJECTS = True
    storage = ObjectStorage(config)
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


def test_get_objects_real_obj_works(storage_fixture_real):
    storage, c = storage_fixture_real
    assert storage.get_objects() is not None
    assert len(storage.get_objects()) == 51  # there are 51 objects exactly
    assert all(storage.get_objects() == storage._objects)
