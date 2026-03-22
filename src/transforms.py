# transforms.py
import math
from src.matrix import Matrix
from src.validator import MatrixValidator

def eigenvalues_2x2(A: Matrix) -> list[float]:
    MatrixValidator.validate_square(A.shape(), "Eigenvalue computation")
    a, b = A.data[0][0], A.data[0][1]
    c, d = A.data[1][0], A.data[1][1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace**2 - 4 * det
    if discriminant < 0:
        raise ValueError(
            "Complex eigenvalues detected. "
            "This matrix represents a pure rotation — no real eigenvectors exist."
        )
    sqrt_disc = math.sqrt(discriminant)
    return [(trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2]


def eigenvector_2x2(A: Matrix, lam: float) -> list[float]:
    a, b = A.data[0][0], A.data[0][1]
    c, d = A.data[1][0], A.data[1][1]
    if b != 0:
        x, y = b, lam - a
    elif c != 0:
        x, y = lam - d, c
    else:
        x, y = 1, 0
    magnitude = math.sqrt(x**2 + y**2)
    return [x / magnitude, y / magnitude]


def eigen_2x2(A: Matrix) -> tuple[list[float], list[list[float]]]:
    lambdas = eigenvalues_2x2(A)   # already validates square
    vectors = [eigenvector_2x2(A, lam) for lam in lambdas]
    return lambdas, vectors