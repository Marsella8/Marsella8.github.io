from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from functools import reduce

Matrix = np.ndarray


def _value(x: Expression) -> Matrix:
    if isinstance(x, Matrix):
        return x
    return x.value()


@dataclass(frozen=True)
class Product:
    args: list[Add | Transpose | Inverse | Matrix]

    def __post_init__(self):
        assert all(not isinstance(arg, Product) for arg in self.args)

    @staticmethod
    def of(a: Expression, b: Expression) -> Product:
        match (a, b):
            case Product(), Product():
                return Product(a.args + b.args)
            case Product(), _:
                return Product(a.args + [b])
            case _, Product():
                return Product([a] + b.args)
        return Product([a, b])

    
    def value(self) -> Matrix:
        head, *tail = self.args
        return reduce(lambda acc, nxt: acc @ _value(nxt), tail, _value(head))

    @property
    def shape(self) -> Matrix:
        m = self.args[0].shape[0]
        n = self.args[-1].shape[-1]
        return (m, n)
    
    def __repr__(self):
        return "(" + "@".join(map(str, self.args)) + ")"

@dataclass(frozen=True)
class Add:
    args: list[Product | Transpose | Inverse | Matrix]  # TODO: should be multiset

    def __post_init__(self):
        assert all(not isinstance(arg, Add) for arg in self.args)

    @staticmethod
    def of(a: Expression, b: Expression) -> Add:
        match (a, b):
            case Add(), Add():
                return Add(a.args + b.args)
            case Add(), _:
                return Add(a.args + [b])
            case _, Add():
                return Add([a] + b.args)
        return Add([a, b])

    def value(self) -> Matrix:
        head, *tail = self.args
        return reduce(lambda acc, nxt: acc + _value(nxt), tail, _value(head))

    @property
    def shape(self) -> Matrix:
        return self.args[0].shape

    def __repr__(self):
        return "+".join(map(str, self.args))


@dataclass(frozen=True)
class Transpose:
    inner: Expression

    def value(self) -> Matrix:
        return _value(self.inner).T  # replaced .value() with _value

    @property
    def shape(self) -> Matrix:
        m, n = self.inner.shape
        return (n, m)

    def __repr__(self):
        return f'{self.inner}.T'


@dataclass(frozen=True)
class Inverse:
    inner: Expression

    def value(self) -> Matrix:
        return np.linalg.inv(_value(self.inner))  # replaced .value() with _value

    @property
    def shape(self) -> Matrix:
        return self.inner.shape

    def __repr__(self):
        return f'{self.inner}.inv'


Expression = Add | Product | Transpose | Inverse | Matrix


@dataclass
class LazyMat:
    expr: Expression

    def __matmul__(self, other: Matrix | LazyMat) -> LazyMat:
        return LazyMat(Product.of(self.expr, _unwrap(other)))

    def __rmatmul__(self, other: Matrix | LazyMat) -> LazyMat:
        return LazyMat(Product.of(_unwrap(other), self.expr))

    def __add__(self, other: Matrix | LazyMat) -> LazyMat:
        return LazyMat(Add.of(self.expr, _unwrap(other)))

    def __radd__(self, other: Matrix | LazyMat) -> LazyMat:
        return LazyMat(Add.of(_unwrap(other), self.expr))

    @property
    def T(self):
        return LazyMat(Transpose(self.expr))

    @property
    def inv(self):
        return LazyMat(Inverse(self.expr))

    def compile(self) -> None:
        from compiler import top_level_compile  # here to avoid circular import
        self.expr = top_level_compile(self.expr)

    def value(self) -> Matrix:
        return _value(self.expr)

    def __array__(self, dtype=None):
        out = self.value()
        return np.asarray(out, dtype=dtype) if dtype is not None else out

def _unwrap(x: LazyMat | Matrix) -> Expression:
    return x if isinstance(x, Matrix) else x.expr
