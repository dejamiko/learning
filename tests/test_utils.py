import concurrent.futures

import numpy as np

from utils import SingletonMeta, get_object_indices, get_bin_representation


def test_singleton_metaclass_works():
    class A(metaclass=SingletonMeta):
        def __init__(self, a):
            self.a = a

    def singleton(a):
        singleton = A(a)
        return singleton.a

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(singleton, p) for p in ["a", "b", "c"]]
        return_value = [f.result() for f in futures]
        assert return_value == ["a", "a", "a"]


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
