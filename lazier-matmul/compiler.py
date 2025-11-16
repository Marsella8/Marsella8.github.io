from lazier_matmul import Expression, Transpose, Inverse, MatMul, Add, LazyMat
from functools import cache
L = LazyMat


@cache
def inner_compile(m: LazyMat) -> tuple[bool, LazyMat]:
    c = inner_compile
    match m:
        # Transpose
        case L(Transpose(L(Transpose(x)))):  # M.T.T = M
            return True, c(x)[1]
        case L(Transpose(L(Inverse(x)))):  # (M.inv).T = (M.T).inv
            return True, c(L(Inverse(L(Transpose(x)))))[1]
        # Inverse
        case L(Inverse(L(Inverse(x)))):  # M.inv.inv = M
            return True, c(x)[1]
        # Add
        case L(Add(x, L(Add(y, z)))):  # A+(B+C) = (A+B)+C
            return True, c(L(Add(L(Add(x, y)), z)))[1]
        case L(Add(L(MatMul(x, y)), L(MatMul(x_, z)))) if x == x_:  # A@B+A@C = A@(B+C)
            return True, c(L(MatMul(x, L(Add(y, z)))))[1]
        case L(Add(L(MatMul(y, x)), L(MatMul(z, x_)))) if x == x_:  # B@A+C@A = (B+C)@A
            return True, c(L(MatMul(L(Add(y, z)), x)))[1]
        # MatMul
        case L(MatMul(x, L(MatMul(y, z)))):  # A@(B@C) = (A@B)@C
            return True, c(L(MatMul(L(MatMul(x, y)), z)))[1]

        # Recursive Compilation
        case L(Transpose(x)):
            c_x, x = c(x)
            if c_x:
                return True, c(L(Transpose(x)))[1]
            return False, L(Transpose(x))

        case L(Inverse(x)):
            c_x, x = c(x)
            if c_x:
                return True, c(L(Inverse(x)))[1]
            return False, L(Inverse(x))
        case L(Add(x, y)):
            (c_x, x), (c_y, y) = c(x), c(y)
            if c_x or c_y:
                return True, c(L(Add(x, y)))[1]
            return False, L(Add(x, y))
        case L(MatMul(x, y)):  # compile recursively
            (c_x, x), (c_y, y) = c(x), c(y)
            if c_x or c_y:
                return True, c(L(MatMul(x, y)))[1]
            return False, L(MatMul(x, y))
        # No valid substitutions found
        case _:
            return False, m


def compiler(m: LazyMat) -> LazyMat:
    _, m = inner_compile(m)
    return m
