"""
validator.py

Pure validation logic for matrix inputs.
No arithmetic. No side effects. Only raises or returns.

Rule: Every method either returns silently (valid) or raises ValueError (invalid).
"""


class MatrixValidator:

    @staticmethod
    def validate_matrix(data: list) -> tuple[int, int]:
        """
        Validate that data is a non-empty, non-jagged 2D list.
        Returns (rows, cols) if valid.
        Raises ValueError if not.
        """
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Matrix data must be a non-empty list.")

        if not isinstance(data[0], list) or len(data[0]) == 0:
            raise ValueError("Matrix rows must be non-empty lists.")

        cols = [len(row) for row in data]
        if len(set(cols)) != 1:
            raise ValueError(
                f"Jagged matrix detected. Row lengths: {cols}. "
                "All rows must have equal number of columns."
            )

        return (len(data), cols[0])

    @staticmethod
    def validate_same_shape(shape1: tuple, shape2: tuple, operation: str) -> None:
        """
        Raises ValueError if two shapes are not identical.
        Used for addition, subtraction, element-wise multiplication.
        """
        if shape1 != shape2:
            raise ValueError(
                f"Shape mismatch for {operation}. "
                f"Got {shape1} and {shape2}. Both matrices must have identical dimensions."
            )

    @staticmethod
    def validate_multipliable(shape1: tuple, shape2: tuple) -> None:
        """
        Raises ValueError if shape1 @ shape2 is not valid.
        Inner dimensions must match: shape1[1] == shape2[0].
        """
        if shape1[1] != shape2[0]:
            raise ValueError(
                f"Cannot multiply: matrix of shape {shape1} "
                f"by matrix of shape {shape2}. "
                f"Inner dimensions must match: {shape1[1]} != {shape2[0]}."
            )

    @staticmethod
    def validate_square(shape: tuple, operation: str) -> None:
        """
        Raises ValueError if the matrix is not square.
        Used for determinant, inverse, trace, etc.
        """
        if shape[0] != shape[1]:
            raise ValueError(
                f"{operation} requires a square matrix. Got shape {shape}."
            )

    @staticmethod
    def validate_row_index(index: int, n_rows: int, label: str = "Row") -> None:
        """
        Raises ValueError if a 1-indexed row index is out of range.
        """
        if not (1 <= index <= n_rows):
            raise ValueError(
                f"{label} index {index} is out of range. "
                f"Matrix has {n_rows} rows. Indices are 1-based."
            )