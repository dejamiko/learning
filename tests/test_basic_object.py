import numpy as np
from pytest import fixture, raises

from config import Config
from playground.basic_object import BasicObject
from tm_utils import (
    Task,
    get_rng,
    SimilarityMeasure,
    ContourSimilarityMeasure,
    ImageEmbeddings,
    ContourImageEmbeddings,
)


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


def test_get_visual_similarity_cosine_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.18602748506878572)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_euclidean_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.EUCLIDEAN
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.2402530733520421)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_manhattan_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.MANHATTAN
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.09090909090909091)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_pearson_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.PEARSON
    obj_0.visible_repr = np.zeros_like(obj_0.visible_repr)
    obj_0.visible_repr[-1] = 1
    obj_0.visible_repr[-2] = 1
    obj_1.visible_repr = np.zeros_like(obj_1.visible_repr)
    obj_1.visible_repr[-1] = 1
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.6666666666666666)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_hausdorff_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    c.SIMILARITY_MEASURE = ContourSimilarityMeasure.HAUSDORFF
    obj_0.visible_repr = [
        [152, 85],
        [151, 86],
        [139, 86],
        [137, 88],
        [137, 118],
        [138, 119],
        [138, 120],
    ]
    obj_1.visible_repr = [
        [52, 85],
        [51, 86],
        [39, 86],
        [37, 88],
        [37, 118],
        [38, 119],
        [38, 120],
        [34, 119],
        [34, 120],
    ]
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.009613589864850152)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_asd_works(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    c.SIMILARITY_MEASURE = ContourSimilarityMeasure.ASD
    obj_0.visible_repr = [
        [152, 85],
        [151, 86],
        [139, 86],
        [137, 88],
        [137, 118],
        [138, 119],
        [138, 120],
    ]
    obj_1.visible_repr = [
        [52, 85],
        [51, 86],
        [39, 86],
        [37, 88],
        [37, 118],
        [38, 119],
        [38, 120],
        [34, 119],
        [34, 120],
    ]
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.010442424171668041)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_other_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = ImageEmbeddings.DOBBE
    with raises(ValueError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert (
        str(e.value) == f"Unknown similarity measure provided `{c.SIMILARITY_MEASURE}`."
    )


@fixture
def object_fixture_multi_dim():
    config = Config()
    np.random.seed(config.SEED)
    obj_0 = BasicObject(
        0,
        config,
        Task.HAMMERING,
        np.zeros((5, config.LATENT_DIM)),
        get_rng(config.SEED),
    )
    np.random.seed(config.SEED)
    obj_1 = BasicObject(
        1, config, Task.HAMMERING, np.ones((5, config.LATENT_DIM)), get_rng(config.SEED)
    )
    return obj_0, obj_1, config


def test_get_visual_similarity_multi_dim_works(object_fixture_multi_dim):
    obj_0, obj_1, c = object_fixture_multi_dim
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.17166119793794804)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_euclidean_multi_dim_works(object_fixture_multi_dim):
    obj_0, obj_1, c = object_fixture_multi_dim
    c.SIMILARITY_MEASURE = SimilarityMeasure.EUCLIDEAN
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.2402530733520421)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_manhattan_multi_dim_works(object_fixture_multi_dim):
    obj_0, obj_1, c = object_fixture_multi_dim
    c.SIMILARITY_MEASURE = SimilarityMeasure.MANHATTAN
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.09090909090909091)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_pearson_multi_dim_works(object_fixture_multi_dim):
    obj_0, obj_1, c = object_fixture_multi_dim
    c.SIMILARITY_MEASURE = SimilarityMeasure.PEARSON
    obj_0.visible_repr = np.zeros_like(obj_0.visible_repr)
    obj_0.visible_repr[:, -1] = np.ones_like(obj_0.visible_repr[:, -1])
    obj_0.visible_repr[:, -2] = np.ones_like(obj_0.visible_repr[:, -2])
    obj_1.visible_repr = np.zeros_like(obj_1.visible_repr)
    obj_1.visible_repr[:, -1] = np.ones_like(obj_1.visible_repr[:, -1])
    obj_1.visible_repr[:, -1] = np.ones_like(obj_1.visible_repr[:, -1])
    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.6666666666666666)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_hausdorff_multi_dim_works(object_fixture_multi_dim):
    obj_0, obj_1, c = object_fixture_multi_dim
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    c.SIMILARITY_MEASURE = ContourSimilarityMeasure.HAUSDORFF
    obj_0.visible_repr = [
        [
            [152, 85],
            [151, 86],
            [139, 86],
            [137, 88],
            [137, 118],
            [138, 119],
            [138, 120],
        ],
        [
            [151, 85],
            [150, 86],
            [138, 86],
            [136, 88],
            [136, 118],
            [137, 119],
            [137, 120],
        ],
        [
            [150, 85],
            [149, 86],
            [137, 86],
            [135, 88],
            [135, 118],
            [136, 119],
            [136, 120],
        ],
        [
            [149, 85],
            [148, 86],
            [136, 86],
            [134, 88],
            [134, 118],
            [135, 119],
            [135, 120],
        ],
        [
            [148, 85],
            [147, 86],
            [135, 86],
            [133, 88],
            [133, 118],
            [134, 119],
            [134, 120],
        ],
    ]
    obj_1.visible_repr = [
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
    ]

    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.009803901870802875)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_asd_multi_dim_works(object_fixture_multi_dim):
    obj_0, obj_1, c = object_fixture_multi_dim
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    c.SIMILARITY_MEASURE = ContourSimilarityMeasure.ASD
    obj_0.visible_repr = [
        [
            [152, 85],
            [151, 86],
            [139, 86],
            [137, 88],
            [137, 118],
            [138, 119],
            [138, 120],
        ],
        [
            [151, 85],
            [150, 86],
            [138, 86],
            [136, 88],
            [136, 118],
            [137, 119],
            [137, 120],
        ],
        [
            [150, 85],
            [149, 86],
            [137, 86],
            [135, 88],
            [135, 118],
            [136, 119],
            [136, 120],
        ],
        [
            [149, 85],
            [148, 86],
            [136, 86],
            [134, 88],
            [134, 118],
            [135, 119],
            [135, 120],
        ],
        [
            [148, 85],
            [147, 86],
            [135, 86],
            [133, 88],
            [133, 118],
            [134, 119],
            [134, 120],
        ],
    ]
    obj_1.visible_repr = [
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
        [
            [52, 85],
            [51, 86],
            [39, 86],
            [37, 88],
            [37, 118],
            [38, 119],
            [38, 120],
            [34, 119],
            [34, 120],
        ],
    ]

    assert np.allclose(obj_0.get_visual_similarity(obj_1), 0.010663930536164556)
    assert np.allclose(obj_0.get_visual_similarity(obj_0), 1)
    assert np.allclose(obj_1.get_visual_similarity(obj_1), 1)


def test_get_visual_similarity_cosine_with_wrong_embeddings_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    with raises(AssertionError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert str(e.value) == (
        f"The ImageEmbeddings provided `{c.IMAGE_EMBEDDINGS}` "
        f"do not work with non-contour similarity measures."
    )


def test_get_visual_similarity_euclidean_with_wrong_embeddings_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.EUCLIDEAN
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    with raises(AssertionError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert str(e.value) == (
        f"The ImageEmbeddings provided `{c.IMAGE_EMBEDDINGS}` "
        f"do not work with non-contour similarity measures."
    )


def test_get_visual_similarity_manhattan_with_wrong_embeddings_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.MANHATTAN
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    with raises(AssertionError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert str(e.value) == (
        f"The ImageEmbeddings provided `{c.IMAGE_EMBEDDINGS}` "
        f"do not work with non-contour similarity measures."
    )


def test_get_visual_similarity_pearson_with_wrong_embeddings_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = SimilarityMeasure.PEARSON
    c.IMAGE_EMBEDDINGS = ContourImageEmbeddings.MASK_RCNN
    with raises(AssertionError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert str(e.value) == (
        f"The ImageEmbeddings provided `{c.IMAGE_EMBEDDINGS}` "
        f"do not work with non-contour similarity measures."
    )


def test_get_visual_similarity_hausdorff_with_wrong_embeddings_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = ContourSimilarityMeasure.HAUSDORFF
    c.IMAGE_EMBEDDINGS = ImageEmbeddings.DOBBE
    with raises(AssertionError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert str(e.value) == (
        f"The ImageEmbeddings provided `{c.IMAGE_EMBEDDINGS}` "
        f"do not work with contour similarity measures."
    )


def test_get_visual_similarity_asd_with_wrong_embeddings_fails(object_fixture):
    obj_0, obj_1, c = object_fixture
    c.SIMILARITY_MEASURE = ContourSimilarityMeasure.ASD
    c.IMAGE_EMBEDDINGS = ImageEmbeddings.DOBBE
    with raises(AssertionError) as e:
        obj_0.get_visual_similarity(obj_1)
    assert str(e.value) == (
        f"The ImageEmbeddings provided `{c.IMAGE_EMBEDDINGS}` "
        f"do not work with contour similarity measures."
    )
