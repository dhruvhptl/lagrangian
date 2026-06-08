from src.utils.reproducibility import set_global_seed
import numpy as np
import random


def test_set_global_seed_numpy_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_random_reproducible():
    set_global_seed(99)
    a = [random.random() for _ in range(5)]
    set_global_seed(99)
    b = [random.random() for _ in range(5)]
    assert a == b
