import numpy as np

from tm_utils import get_object_indices, get_bin_representation


def test_get_object_indices_works():
    selected = [0, 3, 9]
    objects = np.zeros(10)
    for s in selected:
        objects[s] = 1
    assert all(get_object_indices(objects) == selected)


def test_get_bin_representation():
    max_length = 10
    selected = [0, 3, 9]
    objects = np.zeros(max_length)
    for s in selected:
        objects[s] = 1
    assert all(get_bin_representation(selected, max_length) == objects)
