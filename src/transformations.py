from src.vector import Vector
from src.matrix import Matrix
import math

def rotation_matrix(theta: float)->"Matrix":
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return Matrix([[cos_theta,-sin_theta],[sin_theta,cos_theta]])
    
def scale_matrix(sx:int|float, sy:int|float)-> "Matrix":
    return Matrix([[sx,0],[0,sy]])

def reflect_x_matrix()->"Matrix":
    return Matrix([[1,0],[0,-1]])

def reflect_y_matrix()->"Matrix":
    return Matrix([[-1,0],[0,1]])

def reflect_line_matrix(theta:int|float)->"Matrix":
    cos_theta = math.cos(2*theta)
    sin_theta = math.sin(2*theta)
    return Matrix([[cos_theta,sin_theta],[sin_theta,-cos_theta]])

def shear_x_matrix(factor:int|float)->"Matrix":
    return Matrix([[1,factor],[0,1]])

def shear_y_matrix(factor:int|float)->"Matrix":
    return Matrix([[1,0],[factor,1]])

def transform(matrix: Matrix, vector:Vector)->Vector:
    if matrix.shape()[-1] !=vector.shape()[0]:
        raise ValueError("Matrix columns must match vector size for transformation.")
    vector_matrix = vector.to_matrix()
    result_matrix = matrix @ vector_matrix
    return Vector([row[0] for row in result_matrix.data])

def rotate_transform(vector:Vector,theta:float|int)->Vector:
    rotate_matrix = rotation_matrix(theta)
    return transform(rotate_matrix,vector)

def reflect_x(vector:Vector)->Vector:
    reflect_matrix = reflect_x_matrix()
    return transform(reflect_matrix,vector)

def reflect_y(vector:Vector)->Vector:
    reflect_matrix = reflect_y_matrix()
    return transform(reflect_matrix,vector)

def shear_x(vector:Vector,factor:float|int)->Vector:
    shear_matrix = shear_x_matrix(factor)
    return transform(shear_matrix,vector)

def shear_y(vector:Vector,factor:float|int)->Vector:
    shear_matrix = shear_y_matrix(factor)
    return transform(shear_matrix,vector)

def reflect_line(vector:Vector,theta:float|int)->Vector:
    reflect_matrix = reflect_line_matrix(theta)
    return transform(reflect_matrix,vector)

def project_onto(vector1:Vector, vector2:Vector)->Vector:
    v1v2_dot = vector1.dot(vector2)
    v2v2_dot = vector2.dot(vector2)
    if v2v2_dot == 0:
        raise ValueError("Cannot project onto zero vector")
    scalar = v1v2_dot/v2v2_dot
    return vector2 * scalar

