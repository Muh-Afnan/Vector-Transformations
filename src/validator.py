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
    def validate_square(shape: tuple, operation: str) -> None:
        """
        Raises ValueError if the matrix is not square.
        Used for determinant, inverse, trace, etc.
        """
        if shape[0] != shape[1]:
            raise ValueError(
                f"{operation} requires a square matrix. Got shape {shape}."
            )
            