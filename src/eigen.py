# transforms.py
import math
from src.matrix import Matrix
from src.validator import MatrixValidator

def eigenvalues_2x2(A: Matrix) -> list[float]:
    if A.shape() != (2, 2):
        raise ValueError("eigenvalues_2x2 only supports 2x2 matrices")
    MatrixValidator.validate_square(A.shape(), "Eigenvalue computation")
    a, b = A.data[0][0], A.data[0][1]
    c, d = A.data[1][0], A.data[1][1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace**2 - 4 * det
    if discriminant < -1e-9:
        raise ValueError(
            "Complex eigenvalues detected. Real eigenvectors do not exist."
        )
    sqrt_disc = math.sqrt(max(0.0, discriminant))
    return [(trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2]


def eigenvector_2x2(A: Matrix, lam: float) -> Matrix:
    a, b = A.data[0][0], A.data[0][1]
    c, d = A.data[1][0], A.data[1][1]
    if b != 0:
        x, y = b, lam - a
    elif c != 0:
        x, y = lam - d, c
    else:
        if lam == a:
            x, y = 1, 0
        else:
            x, y = 0, 1
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        raise ValueError("Zero vector encountered")
    return Matrix([[x / magnitude], [y / magnitude]])

def eigen_2x2(A: Matrix) -> tuple[list[float], list[list[float]]]:
    lambdas = eigenvalues_2x2(A)   # already validates square
    vectors = [eigenvector_2x2(A, lam) for lam in lambdas]
    return lambdas, vectors