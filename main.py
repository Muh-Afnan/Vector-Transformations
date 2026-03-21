import copy
from src.matrix import Matrix
from src.validator import MatrixValidator


A = Matrix([[1, 2], [3, 4]])

def calculate_eigen_values(matrix:Matrix)->list[int]:
    shape = matrix.shape()
    if MatrixValidator.validate_square(shape):
        counter = 0
        for row in matrix:
            matrix[counter][counter].concat("-x")
            counter +=1
    matrix        