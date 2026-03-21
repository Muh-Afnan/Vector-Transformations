import copy
from src.validator import MatrixValidator

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
    # Matrix property checks
    # ------------------------------------------------------------------

    def is_square(self) -> bool:
        return self.rows == self.cols

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