"""
matrix.py

The core Matrix class. Represents a matrix as an object.

Design decisions:
- All dunder methods return Matrix instances, enabling chaining: (A + B) @ C
- All operations are non-mutating: every method returns a new Matrix
- Raw list data is accessible via .data for interop with external code
- Validation is delegated to MatrixValidator — Matrix itself contains no validation logic
- Arithmetic operations (determinant, inverse, etc.) live in operations.py
  to keep this file focused on the object model

Usage:
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A + B       # Matrix([[6, 8], [10, 12]])
    D = A @ B       # Matrix([[19, 22], [43, 50]])
    print(A)        # formatted output
"""

import copy
from src.validator import MatrixValidator
import math


class Matrix:

    def __init__(self, data: list[list[int | float]]):
        """
        Initialise a Matrix from a 2D list.
        Raises ValueError if data is empty, jagged, or not a 2D list.
        """
        self.rows, self.cols = MatrixValidator.validate_matrix(data)
        self.data = copy.deepcopy(data)  # defensive copy — caller mutation won't affect us

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def shape(self) -> tuple[int, int]:
        """Return (rows, cols)."""
        return (self.rows, self.cols)

    # ------------------------------------------------------------------
    # Dunder methods — all return Matrix instances
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        rows = ['[' + ', '.join(str(x) for x in row) + ']' for row in self.data]
        return 'Matrix([\n  ' + ',\n  '.join(rows) + '\n])'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape() != other.shape():
            return False
        return all(
            abs(v1 - v2) < 1e-9
            for r1, r2 in zip(self.data, other.data)
            for v1, v2 in zip(r1, r2)
        )

    def __add__(self, other: "Matrix") -> "Matrix":
        MatrixValidator.validate_same_shape(self.shape(), other.shape(), "addition")
        result = [
            [a + b for a, b in zip(r1, r2)]
            for r1, r2 in zip(self.data, other.data)
        ]
        return Matrix(result)

    def __sub__(self, other: "Matrix") -> "Matrix":
        MatrixValidator.validate_same_shape(self.shape(), other.shape(), "subtraction")
        result = [
            [a - b for a, b in zip(r1, r2)]
            for r1, r2 in zip(self.data, other.data)
        ]
        return Matrix(result)

    def __mul__(self, other: "int | float | Matrix") -> "Matrix":
        """
        A * scalar  → scalar multiplication
        A * B       → element-wise (Hadamard) multiplication
        For dot product use A @ B.
        """
        if isinstance(other, (int, float)):
            return Matrix([[item * other for item in row] for row in self.data])
        if isinstance(other, Matrix):
            MatrixValidator.validate_same_shape(self.shape(), other.shape(), "element-wise multiplication")
            result = [
                [a * b for a, b in zip(r1, r2)]
                for r1, r2 in zip(self.data, other.data)
            ]
            return Matrix(result)
        return NotImplemented

    def __rmul__(self, scalar: "int | float") -> "Matrix":
        """scalar * A"""
        return self.__mul__(scalar)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """A @ B — standard matrix (dot) product."""
        MatrixValidator.validate_multipliable(self.shape(), other.shape())
        other_T = list(zip(*other.data))
        result = [
            [sum(a * b for a, b in zip(row, col)) for col in other_T]
            for row in self.data
        ]
        return Matrix(result)

    # ------------------------------------------------------------------
    # Core matrix operations
    # ------------------------------------------------------------------

    def transpose(self) -> "Matrix":
        """Return the transpose of this matrix."""
        return Matrix([list(row) for row in zip(*self.data)])

    def trace(self) -> int | float:
        """
        Return the sum of diagonal elements.
        Raises ValueError if matrix is not square.
        """
        MatrixValidator.validate_square(self.shape(), "Trace")
        return sum(self.data[i][i] for i in range(self.rows))

    # ------------------------------------------------------------------
    # Row operations — all return a new Matrix, never mutate self
    # ------------------------------------------------------------------

    def swap_rows(self, r1: int, r2: int) -> "Matrix":
        """
        Return new Matrix with rows r1 and r2 swapped.
        Rows are 1-indexed.
        """
        MatrixValidator.validate_row_index(r1, self.rows, "r1")
        MatrixValidator.validate_row_index(r2, self.rows, "r2")
        new_data = copy.deepcopy(self.data)
        new_data[r1 - 1], new_data[r2 - 1] = new_data[r2 - 1], new_data[r1 - 1]
        return Matrix(new_data)

    def add_rows(self, target_row: int, source_row: int) -> "Matrix":
        """
        Return new Matrix with target_row replaced by target_row + source_row.
        Rows are 1-indexed.
        """
        MatrixValidator.validate_row_index(target_row, self.rows, "target_row")
        MatrixValidator.validate_row_index(source_row, self.rows, "source_row")
        new_data = copy.deepcopy(self.data)
        new_data[target_row - 1] = [
            a + b for a, b in zip(self.data[target_row - 1], self.data[source_row - 1])
        ]
        return Matrix(new_data)

    def scale_row(self, row: int, factor: int | float) -> "Matrix":
        """
        Return new Matrix with the given row multiplied by factor.
        Rows are 1-indexed.
        """
        MatrixValidator.validate_row_index(row, self.rows, "row")
        new_data = copy.deepcopy(self.data)
        new_data[row - 1] = [item * factor for item in self.data[row - 1]]
        return Matrix(new_data)

    # ------------------------------------------------------------------
    # Matrix property checks
    # ------------------------------------------------------------------

    def is_square(self) -> bool:
        return self.rows == self.cols

    def is_symmetric(self) -> bool:
        MatrixValidator.validate_square(self.shape(), "Symmetry check")
        return self == self.transpose()

    def is_skew_symmetric(self) -> bool:
        MatrixValidator.validate_square(self.shape(), "Skew-symmetry check")
        T = self.transpose()
        return all(
            self.data[i][j] == -T.data[i][j]
            for i in range(self.rows)
            for j in range(self.cols)
        )

    def is_diagonal(self) -> bool:
        MatrixValidator.validate_square(self.shape(), "Diagonal check")
        return all(
            self.data[i][j] == 0
            for i in range(self.rows)
            for j in range(self.cols)
            if i != j
        )

    def is_identity(self) -> bool:
        MatrixValidator.validate_square(self.shape(), "Identity check")
        return all(
            self.data[i][j] == (1 if i == j else 0)
            for i in range(self.rows)
            for j in range(self.cols)
        )

    def is_zero(self) -> bool:
        return all(item == 0 for row in self.data for item in row)
        