from __future__ import annotations
import numpy as np
from functools import cached_property
from dataclasses import dataclass


@dataclass(frozen=True)
class Matrix:
    mat: np.ndarray

    @cached_property
    def value(self) -> np.ndarray:
        return self.mat

    def __hash__(self) -> int:
        return id(self.mat)


@dataclass(frozen=True)
class MatMul:
    x: LazyMat
    y: LazyMat

    @cached_property
    def value(self) -> Matrix:
        return self.x.value @ self.y.value


@dataclass(frozen=True)
class Add:
    x: LazyMat
    y: LazyMat

    @cached_property
    def value(self) -> Matrix:
        return self.x.value + self.y.value


@dataclass(frozen=True)
class Transpose:
    mat: LazyMat

    @cached_property
    def value(self) -> Matrix:
        return self.mat.value.T


@dataclass(frozen=True)
class Inverse:
    mat: LazyMat

    @cached_property
    def value(self) -> Matrix:
        return np.linalg.inv(self.mat.value)


Expression = MatMul | Add | Transpose | Inverse | Matrix


@dataclass(frozen=True)
class LazyMat:
    expr: Expression

    def __matmul__(self, other: LazyMat) -> LazyMat:
        return LazyMat(MatMul(self, other))

    def __rmatmul__(self, other: LazyMat) -> LazyMat:
        return self @ other

    def __add__(self, other: LazyMat) -> LazyMat:
        return LazyMat(Add(self, other))

    def __radd__(self, other: LazyMat) -> LazyMat:
        return self + other

    @property
    def T(self) -> LazyMat:
        return LazyMat(Transpose(self))

    @property
    def inv(self) -> LazyMat:
        return LazyMat(Inverse(self))

    @cached_property
    def value(self) -> Matrix:
        return self.expr.value

    def __eq__(self, other: LazyMat) -> bool:
        match (self.expr, other.expr):
            case Matrix(x), Matrix(y):
                return np.allclose(x, y)
        return self.expr == other.expr

    @cached_property
    def cost(self) -> int:
        seen: set[LazyMat] = set()

        def visit(m: LazyMat) -> int:
            if m in seen:
                return 0
            seen.add(m)
            match m.expr:
                case Matrix():
                    return 0
                case MatMul(x, y) | Add(x, y):
                    return visit(x) + visit(y) + 1
                case Transpose(x) | Inverse(x):
                    return visit(x) + 1

        return visit(self)
    
    def __repr__(self) -> str:
        match self.expr:
            case Matrix():
                return str(id(self.expr.mat))[-3:]
            case MatMul(x, y):
                return f"({x} @ {y})"
            case Add(x, y):
                return f"({x} + {y})"
            case Transpose(x):
                return f"{x}.T"
            case Inverse(x):
                return f"{x}.inv"
