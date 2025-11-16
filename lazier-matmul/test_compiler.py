from lazier_matmul import LazyMat, Matrix, MatMul, Add, Transpose, Inverse
from compiler import compiler as compile
import numpy as np

A = LazyMat(Matrix(np.array([[1, 2], [3, 4]])))
B = LazyMat(Matrix(np.array([[5, 6], [7, 8]])))
C = LazyMat(Matrix(np.array([[9, 10], [11, 12]])))
D = LazyMat(Matrix(np.array([[13, 14], [15, 16]])))


def test_transpose_transpose():
    X = A.T.T
    cX = compile(X)
    assert cX == A


def test_inverse_inverse():
    X = A.inv.inv
    cX = compile(X)
    assert cX == A


def test_bubble_up_transpose():
    X = A.inv.T
    cX = compile(X)
    assert cX == A.T.inv
    

def test_composite_transpose_and_inverse():
    X = A.inv.T.inv.T
    cX = compile(X)
    assert cX == A


def test_matmul_associativity():
    X = A @ (B @ C)
    cX = compile(X)
    assert cX == (A @ B) @ C


def test_add_associativity():
    X = A + (B + C)
    cX = compile(X)
    assert cX == (A + B) + C


def test_distribute_matmul_over_add_left():
    X = (A @ B) + (A @   C)
    cX = compile(X)
    assert cX == A @ (B + C)


def test_matmul_associativity_deep():
    X = A @ (B @ (C @ A))
    cX = compile(X)
    assert cX == ((A @ B) @ C) @ A


def test_add_associativity_deep():
    X = A + (B + (C + A))
    cX = compile(X)
    assert cX == ((A + B) + C) + A


def test_left_distribution_three_terms():
    X = (A @ B) + ((A @ C) + (A @ A))
    cX = compile(X)
    assert cX == A @ ((B + C) + A)


def test_nested_distribution_same_left():
    X = (A @ (B + C)) + (A @ (A + B))
    cX = compile(X)
    assert cX == A @ (((B + C) + A) + B)


def test_double_matmul_add_combo():
    X = ((A @ B) + (A @ C)) + ((A @ A) + (A @ B))
    cX = compile(X)
    assert cX == A @ (((B + C) + A) + B)


def test_nested_matmul_add():
    X = (A @ (B @ C)) + (A @ (B @ A))
    cX = compile(X)
    assert cX == (A @ B) @ (C + A)


def test_two_fold_distribution():
    X = A @ C + A @ D + B @ C + B @ D
    cX = compile(X)
    assert cX == (A + B) @ (C + D), "This test will fail!"
