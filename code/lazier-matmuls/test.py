import numpy as np

from lazier_matmul import LazyMat, Product, Add, Transpose, Inverse, Matrix


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def mat(n: int = 2) -> LazyMat:
    """Return a random positive‑definite matrix wrapped as *LazyMat*."""
    rng = np.random.default_rng()
    return LazyMat(rng.standard_normal((n, n)) + n * np.eye(n))


def I(n: int = 2) -> LazyMat:  # noqa: E743 (shadow built‑in "I")
    return LazyMat(np.eye(n))


def assert_allclose(a: Matrix, b: Matrix, *, atol: float = 1e-10) -> None:
    assert np.allclose(
        a, b, atol=atol, rtol=0.0
    ), f"Matrices differ beyond {atol} absolute tolerance"



def test_double_transpose_simplifies():
    A = mat()
    input = A.T.T
    input.compile()
    correct = A

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Matrix)



def test_transpose_of_inverse_swaps():
    A = mat()
    input = A.inv.T
    input.compile()
    correct = A.T.inv

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Inverse)
    assert isinstance(input.expr.inner, Transpose)



def test_double_inverse_simplifies():
    A = mat()
    input = A.inv.inv
    input.compile()
    correct = A

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Matrix)



def test_add_single_element_eliminates_wrapper():
    A = mat()
    input = LazyMat(Add([A.expr]))
    input.compile()
    correct = A

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Matrix)



def test_product_single_element_eliminates_wrapper():
    A = mat()
    input = LazyMat(Product([A.expr]))
    input.compile()
    correct = A

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Matrix)



def test_common_prefix_factoring():
    A, B, C = mat(), mat(), mat()

    input = A @ B + A @ C
    input.compile()
    print(input)
    correct = A @ (B + C)

    assert_allclose(input.value(), correct.value())
    assert len(input.expr.args) == 2


def test_three_term_common_prefix_factoring():
    A, B, C = mat(), mat(), mat()
    eye = I()

    input = A @ B @ C + A @ B + A
    input.compile()
    correct = A @ (B @ (C + eye) + eye)

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Product)
    assert len(input.expr.args) == 2



def test_mixed_double_transpose_in_product():
    A, B, C = mat(), mat(), mat()

    input = A.T.T @ B.inv.T.T @ C.inv.inv.T.T
    input.compile()

    correct = A @ B.inv @ C

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Product)
    assert len(input.expr.args) == 3
    assert [type(m) for m in correct.expr.args] == [Matrix, Inverse, Matrix]


def test_prefix_factoring_after_inner_simplification():
    A, B, C = mat(), mat(), mat()

    input = A.T.T @ B + A @ C
    input.compile()
    correct = A @ (B + C)

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Product)
    assert len(input.expr.args) == 2


def test_complex_composition_rewrites_to_identity():
    A = mat()
    input = A.T.inv.T.inv
    input.compile()
    correct = A

    assert_allclose(input.value(), correct.value())
    assert isinstance(input.expr, Matrix)

def test_large_expression_simplification():
    A, B, C, D = mat(), mat(), mat(), mat()
    eye = I()

    input_ = ((A.T.T @ B.inv.inv) @ C @ D.inv.T).inv.T +  (A @ eye @ B) + C.T.T
    print(input_)
    input_.compile()
    print(input_)

    correct = ((A@B) @ C @ D.inv.T).inv.T + (A@B) + C

    assert_allclose(input_.value(), correct.value())
    assert isinstance(input_.expr, Add)
    assert len(input_.expr.args) == 3

#TODO: main problems are around how we put together Product and Add, namely the flattening part. Ideally change of to be a *args function and have that if the length is 1 then we simply turn into a Expression
#TODO: remake all the tests to check for structure