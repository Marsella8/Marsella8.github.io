# Transformations:
# Product -> Optimized Parenthesized Product (which boils down to a tree of binary muls)
# Product and Add -> Associate
# Transpose -> Elide Double Transpose X
# Inverse -> Elide double inverse X
# A.inv @ b.inv -> (B@A).inv
# same for transpose
# pull out the T, inside the inv (so we have more changes to double elide)  X
# A.inv @ A become the identity.
# Identity Simplifies
# Zero simplifies

from lazier_matmul import Product, Add, Transpose, Inverse, Expression, Matrix
from typing import Iterable, Any
import numpy as np
from multimethod import multimethod
from functools import reduce


def _get_only(x: Iterable[Any]) -> Any:
    x = list(x)
    assert len(x) == 1
    return x[0]


def group_by_common_prefix(
    args: list[Expression],
) -> list[tuple[Expression, list[Expression]]]:
    normalized = [
        expr if isinstance(expr, Product) else Product([expr, np.eye(expr.shape[0])])
        for expr in args
    ]

    groups: list[tuple[Expression, list[Expression]]] = []
    for prod in normalized:
        prefix, *rest = prod.args
        for existing_prefix, suffixes in groups:
            if existing_prefix is prefix:
                suffixes.append(Product(rest))
                break
        else:
            groups.append((prefix, [Product(rest)]))
    return groups


@multimethod
def compile(x: Transpose) -> Expression:
    inner = compile(x.inner)

    if isinstance(inner, Transpose):
        return inner.inner  # M.T.T = M
    if isinstance(inner, Inverse):
        return Inverse(Transpose(inner.inner))  # (M.inv).T = (M.T).inv

    return Transpose(inner)


@multimethod
def compile(x: Inverse) -> Expression:
    inner = compile(x.inner)

    if isinstance(inner, Inverse):
        return inner.inner  # M.inv.inv = M

    return Inverse(inner)


@multimethod
def compile(x: Add) -> Expression:
    compiled_args = [compile(arg) for arg in x.args]
    if len(compiled_args) == 1:
        return _get_only(compiled_args)
    new_add = Add(compiled_args)

    grouped = group_by_common_prefix(new_add.args)
    factored_args = [
        Product.of(prefix, reduce(Add.of, suffixes)) for prefix, suffixes in grouped
    ]
    return reduce(Add.of, factored_args)


@multimethod
def compile(x: Product) -> Expression:
    compiled_args = [compile(arg) for arg in x.args]
    if len(compiled_args) == 1:
        return _get_only(compiled_args)
    return reduce(Product.of, compiled_args)

@multimethod
def compile(x: Matrix) -> Matrix:
    return x

def top_level_compile(x: Expression) -> Expression:
    for _ in range(100):
        x = compile(x)
    return x
