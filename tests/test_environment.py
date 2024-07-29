from pprint import pprint

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
    config.SIMILARITY_THRESHOLD = 0.85
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
        5: {(1, np.float64(0.776313538644145))},
        6: {(1, np.float64(0.855309749838852))},
        8: {(0, np.float64(0.930177692922736)), (3, np.float64(0.7282439166900824))},
        9: {(1, np.float64(0.7503605171583682))},
        10: {(0, np.float64(0.8775077733175265))},
        12: {(1, np.float64(0.7649813794676803))},
        14: {(1, np.float64(0.8527170277972236))},
        19: {(0, np.float64(0.9329451269068241))},
        20: {(1, np.float64(0.7596890367351331))},
        21: {(1, np.float64(0.8293254215424158))},
        22: {(0, np.float64(0.7785829242873963))},
        24: {(0, np.float64(0.751171782184369))},
        26: {(3, np.float64(0.7714644619957162)), (0, np.float64(0.8165180548000925))},
        29: {(0, np.float64(0.7781793350760904))},
        32: {(1, np.float64(0.8331469325332468))},
        33: {(0, np.float64(0.8344154280741761))},
        36: {(0, np.float64(0.7540723173712294))},
        38: {(3, np.float64(0.7237675144692239)), (0, np.float64(0.8569988338843554))},
        41: {(0, np.float64(0.8245022400092635))},
        42: {(0, np.float64(0.8580727057109294)), (3, np.float64(0.8823795549140315))},
        43: {(1, np.float64(0.7838084815049353))},
        47: {(1, np.float64(0.8342623704464908))},
        48: {(0, np.float64(0.7322019784439039))},
        49: {(0, np.float64(0.7740250570647553))},
        50: {(0, np.float64(0.7354198210170313))},
    }

    actual = env.get_reachable_object_indices(selected)
    pprint(actual)
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
