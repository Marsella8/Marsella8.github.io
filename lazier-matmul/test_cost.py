from lazier_matmul import LazyMat, Matrix
import numpy as np

A = LazyMat(Matrix(np.array([[1, 2], [3, 4]])))
B = LazyMat(Matrix(np.array([[5, 6], [7, 8]])))
C = LazyMat(Matrix(np.array([[9, 10], [11, 12]])))
D = LazyMat(Matrix(np.array([[2, 0], [1, 2]])))


def test_basic_cost():
    assert A.cost == 0
    assert (A @ B).cost == 1
    assert (A + B).cost == 1
    assert A.T.cost == 1
    assert A.inv.cost == 1
    assert (A @ B @ C).cost == 2
    assert (A + B + C).cost == 2


def test_diamond_pattern():
    shared = A @ B
    result = shared + shared
    assert shared.cost == 1
    assert result.cost == 2


def test_multiple_diamond_pattern():
    shared1 = A @ B
    shared2 = B @ C
    result = (shared1 + shared2) @ (shared1 @ shared2)
    assert result.cost == 5


def test_deep_sharing():
    shared = A @ B
    shared_squared = shared @ shared
    result = shared_squared + shared_squared
    assert result.cost == 3


def test_complex_dag():
    shared1 = A @ B
    shared2 = C @ D
    intermediate = shared1 + shared2
    result = (intermediate @ shared1) + (intermediate @ shared2)
    assert result.cost == 6


def test_transpose_and_inverse_sharing():
    shared_T = A.T
    result = (shared_T @ shared_T) + shared_T
    assert result.cost == 3


def test_wide_diamond():
    shared = A @ B
    result = shared + shared + shared + shared
    assert result.cost == 4


def test_nested_operations_with_sharing():
    shared = A @ B
    result = (shared.T @ shared) + (shared @ shared.inv)
    assert result.cost == 6
