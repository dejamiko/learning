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
    config.OBJ_NUM = 100
    config.USE_REAL_OBJECTS = False
    config.SIMILARITY_THRESHOLD = 0.85
    env = Environment(config)
    return env, config


def test_get_objects_works(env_fixture):
    env, config = env_fixture
    assert len(env.get_objects()) == config.OBJ_NUM


def test_get_visual_similarity_works(env_fixture):
    env, config = env_fixture
    assert env.get_visual_similarity(0, 1) == env.storage.get_visual_similarity(0, 1)
    assert env.get_visual_similarity(0, 1) == 0.0
    assert env.get_visual_similarity(0, 2) == 0.6297178434321439
    assert env.get_visual_similarity(0, 4) == 0.7615928478640059
    assert env.get_visual_similarity(0, 1, (2, 0)) == 0.0
    assert env.get_visual_similarity(0, 2, (2, 0)) == 1.0
    assert env.get_visual_similarity(0, 4, (2, 0)) == 1.0
    assert env.get_visual_similarity(0, 1, (1, -0.1)) == 0.0
    assert np.allclose(env.get_visual_similarity(0, 2, (1, -0.1)), 0.5297178434321439)
    assert np.allclose(env.get_visual_similarity(0, 4, (1, -0.1)), 0.6615928478640059)


def test_get_reachable_object_indices_works(env_fixture):
    env, config = env_fixture
    selected = [0, 1, 3]
    expected = {
        0: {(0, np.float64(0.999999999982103))},
        1: {(1, np.float64(0.9999999999722714))},
        3: {(3, np.float64(0.9999999999759837))},
        4: {(0, np.float64(0.7615928478640059))},
        5: {(1, np.float64(0.844567867513219))},
        7: {(3, np.float64(0.9274857313887366))},
        8: {(0, np.float64(0.7221081023556257))},
        9: {(3, np.float64(0.8500786667068724))},
        11: {(1, np.float64(0.7757792936554706))},
        13: {(3, np.float64(0.8197509930153577))},
        14: {(0, np.float64(0.7214295644739119))},
        16: {(3, np.float64(0.9011047701956557))},
        17: {(3, np.float64(0.7582065102761495))},
        18: {(3, np.float64(0.930504283849278))},
        19: {(0, np.float64(0.6982230245771024))},
        20: {(1, np.float64(0.7420473032534884))},
        26: {(1, np.float64(0.7754640015431565))},
        27: {(3, np.float64(0.7952812150506772))},
        29: {(0, np.float64(0.7605709163786001))},
        30: {(3, np.float64(0.9323678631157303))},
        33: {(1, np.float64(0.7203230277754488))},
        35: {(3, np.float64(0.783492427324147))},
        40: {(0, np.float64(0.6965443581843246))},
        41: {(3, np.float64(0.7718289604588451))},
        42: {(0, np.float64(0.7460315568336592))},
        45: {(1, np.float64(0.7129841615927403))},
        49: {(0, np.float64(0.7530483265755431))},
        51: {(3, np.float64(0.7074790395538023))},
        54: {(0, np.float64(0.7011657323944226))},
        55: {(1, np.float64(0.7429309108523))},
        59: {(3, np.float64(0.739226220710281))},
        63: {(0, np.float64(0.7244502355959824))},
        64: {(1, np.float64(0.6920628740164759))},
        66: {(1, np.float64(0.7227763461119636))},
        67: {(1, np.float64(0.7334059795398483))},
        70: {(1, np.float64(0.8838472594808968))},
        71: {(3, np.float64(0.8141561136799546))},
        73: {(1, np.float64(0.836897931353429))},
        78: {(1, np.float64(0.8312504316766411))},
        79: {(0, np.float64(0.7401230114630276))},
        84: {(0, np.float64(0.7623941708484167))},
        86: {(0, np.float64(0.8673067209720565))},
        87: {(3, np.float64(0.765004164767046))},
        89: {(0, np.float64(0.8569205530130777))},
        90: {(1, np.float64(0.7640549759041161))},
        97: {(3, np.float64(0.9249051356359771))},
        99: {(0, np.float64(0.7268830822622099))},
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
    print(answers)
    assert answers == [
        np.True_,
        np.True_,
        np.True_,
        np.True_,
        np.False_,
        np.False_,
        np.True_,
        np.True_,
        np.True_,
        np.False_,
    ]


def test_evaluate_selection_transfer_based_works(env_fixture):
    env, config = env_fixture
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    assert env.evaluate_selection_transfer_based(selected) == 47


def test_evaluate_selection_similarity_based_works(env_fixture):
    env, config = env_fixture
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    assert env.evaluate_selection_visual_similarity_based(selected) == 12


def test_evaluate_selection_similarity_based_real_value_works(env_fixture):
    env, config = env_fixture
    config.SUCCESS_RATE_BOOLEAN = False
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    assert np.allclose(
        env.evaluate_selection_visual_similarity_based(selected), 33.63231293713767
    )


def test_evaluate_selection_similarity_based_other_threshold_works(env_fixture):
    env, config = env_fixture
    selected = np.zeros(config.OBJ_NUM)
    selected[0] = 1
    selected[1] = 1
    selected[3] = 1

    env.update_visual_sim_threshold(0.5)

    assert env.evaluate_selection_visual_similarity_based(selected) == 83


def test_update_visual_sim_threshold(env_fixture):
    env, config = env_fixture
    assert env.visual_sim_threshold == 0.85
    env.update_visual_sim_threshold(0.123)
    assert env.visual_sim_threshold == 0.123


def test_update_visual_similarities(env_fixture):
    env, config = env_fixture
    assert env.similarity_matrix[0, 1] == 0.0
    assert env.similarity_matrix[0, 2] == 0.6297178434321439
    assert env.similarity_matrix[0, 4] == 0.7615928478640059

    env.update_visual_similarities((2, 0))

    assert env.similarity_matrix[0, 1] == 0.0
    assert env.similarity_matrix[0, 2] == 1.0
    assert env.similarity_matrix[0, 4] == 1.0

    env.update_visual_similarities((1, -0.1))

    assert env.similarity_matrix[0, 1] == 0.0
    assert np.allclose(env.similarity_matrix[0, 2], 0.5297178434321439)
    assert np.allclose(env.similarity_matrix[0, 4], 0.6615928478640059)
