from typing import Callable, Any
from functools import reduce
import random
import numpy as np
from lazier_matmul import LazyMat, Matrix, Transpose, Inverse, MatMul, Add
from compiler import compiler

np.random.seed(0)
random.seed(0)

RANDOM_MATRICES = [np.random.randn(2,2) for _ in range(1)]

def random_lazy_mat(n: int) -> LazyMat:
    match n:
        case 0:
            return LazyMat(Matrix(random.choice(RANDOM_MATRICES)))
        case 1:
            op = random.choice([Transpose, Inverse])
            return LazyMat(op(random_lazy_mat(0)))
        case _:
            op = random.choice([Transpose, Inverse, MatMul, Add])
            if op in (MatMul, Add):
                k = random.randint(0, n-1)
                return LazyMat(op(random_lazy_mat(k), random_lazy_mat(n-k-1)))
            else:
                return LazyMat(op(random_lazy_mat(n-1)))

if __name__ == "__main__":
    mat = random_lazy_mat(200)
    print(mat.cost)
    print(compiler(mat).cost)
    print(compiler(mat))
    