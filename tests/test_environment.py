import numpy as np
from pytest import fixture

from config import Config
from playground.environment import Environment


def test_environment_init_works():
    config = Config()
    _ = Environment(config)


@fixture
def env_fixture():
    config = Config()
    config.USE_REAL_OBJECTS = False
    env = Environment(config)
    return env, config


def test_get_objects_works(env_fixture):
    env, config = env_fixture
    assert all(env.get_objects() == env.storage.get_objects())


def test_get_visual_similarity_works(env_fixture):
    env, config = env_fixture
    assert env.get_visual_similarity(0, 1) == env.storage.get_visual_similarity(0, 1)


def test_get_reachable_object_indices_works(env_fixture):
    env, config = env_fixture
    selected = [0, 1, 3]
    expected = {
        0: {(0, np.float64(0.9999999999935263))},
        1: {(1, np.float64(0.9999999999802108))},
        3: {(3, np.float64(0.9999999999674792))},
        6: {(1, np.float64(0.855309749838852))},
        8: {(0, np.float64(0.930177692922736))},
        10: {(0, np.float64(0.8775077733175265))},
        19: {(0, np.float64(0.9329451269068241))},
        38: {(0, np.float64(0.8569988338843554))},
        42: {(0, np.float64(0.8580727057109294)), (3, np.float64(0.8823795549140315))},
    }

    actual = env.get_reachable_object_indices(selected)
    print(actual)
    assert len(expected) == len(actual)
    assert expected.keys() == actual.keys()
    for k in expected.keys():
        assert k in actual
        assert expected[k] == actual[k]


def test_get_transfer_success_works(env_fixture):
    env, config = env_fixture
    config.SUCCESS_RATE_BOOLEAN = False
    answers = []
    for _ in range(10):
        answers.append(env.get_transfer_success(0, 42))
    assert answers == [
        np.False_,
        np.True_,
        np.True_,
        np.True_,
        np.True_,
        np.True_,
        np.True_,
        np.True_,
        np.True_,
        np.True_,
    ]


def test_evaluate_selection_transfer_based_works(env_fixture):
    env, config = env_fixture
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    assert env.evaluate_selection_transfer_based(selected) == 29


def test_evaluate_selection_similarity_based_works(env_fixture):
    env, config = env_fixture
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    assert env.evaluate_selection_visual_similarity_based(selected) == 10


def test_evaluate_selection_similarity_based_other_threshold_works(env_fixture):
    env, config = env_fixture
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    assert env.evaluate_selection_visual_similarity_based(selected, 0.5) == 40
