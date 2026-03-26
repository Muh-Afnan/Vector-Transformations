from src.vector import Vector
from src.matrix import Matrix



def transform(matrix: Matrix, vector:Vector)->Vector:
    if matrix.shape()[-1] !=vector.shape()[0]:
        raise ValueError("Matrix columns must match vector size for transformation.")
    vector_matrix = vector.to_matrix()
    result_matrix = matrix @ vector_matrix
    return Vector([row[0] for row in result_matrix.data])