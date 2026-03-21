"""
operations.py

Advanced mathematical operations on Matrix objects.

Why separate from matrix.py?
- Matrix class handles the object model: construction, operators, properties
- This module handles heavier mathematical procedures: determinant, inverse,
  cofactor, adjoint — operations that involve recursion or multi-step algorithms
- Keeps matrix.py focused and readable
- These functions can be imported and used independently in future modules
  (e.g., a decomposition module, a solver module)

All functions are pure — they take Matrix objects and return new Matrix objects
or scalars. Nothing is mutated.
"""

from src.matrix import Matrix
from src.validator import MatrixValidator


def determinant(matrix:Matrix)->int | float