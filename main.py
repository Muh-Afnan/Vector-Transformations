# import math

# from src.eigen import eigen_2x2
# from src.matrix import Matrix
# from src.transformations import (
#     project_onto,
#     reflect_line,
#     reflect_x,
#     reflect_x_matrix,
#     reflect_y,
#     reflect_y_matrix,
#     rotate_transform,
#     rotation_matrix,
#     scale_matrix,
#     shear_x,
#     shear_x_matrix,
#     shear_y,
#     shear_y_matrix,
#     transform,
# )
# from src.vector import Vector


# def run_vector_visualization_demo() -> None:
#     print("\n=== Vector Visualization Demo ===")
#     v = Vector([2, 3])
#     print(f"Original vector: {v}")
#     v.visualize(label="Original Vector")


# def run_matrix_factory_demo() -> None:
#     print("\n=== Matrix Factory Demo ===")
#     theta = math.pi / 4
#     print(f"rotation_matrix(pi/4):\n{rotation_matrix(theta)}")
#     print(f"scale_matrix(2, 0.5):\n{scale_matrix(2, 0.5)}")
#     print(f"reflect_x_matrix():\n{reflect_x_matrix()}")
#     print(f"reflect_y_matrix():\n{reflect_y_matrix()}")
#     print(f"shear_x_matrix(1.5):\n{shear_x_matrix(1.5)}")
#     print(f"shear_y_matrix(1.5):\n{shear_y_matrix(1.5)}")


# def run_transform_demo() -> None:
#     print("\n=== Transformation Demo ===")
#     v = Vector([1, 2])

#     matrix_result = transform(scale_matrix(2, 1.5), v, visualize=True, title="General Transform")
#     print(f"transform(scale_matrix(2, 1.5), {v}) -> {matrix_result}")

#     rotated = rotate_transform(v, math.pi / 6, visualize=True, title="Rotate by 30 degrees")
#     print(f"rotate_transform({v}, pi/6) -> {rotated}")

#     rx = reflect_x(v, visualize=True, title="Reflect Across X")
#     print(f"reflect_x({v}) -> {rx}")

#     ry = reflect_y(v, visualize=True, title="Reflect Across Y")
#     print(f"reflect_y({v}) -> {ry}")

#     sx = shear_x(v, 1.25, visualize=True, title="Shear X by 1.25")
#     print(f"shear_x({v}, 1.25) -> {sx}")

#     sy = shear_y(v, 1.25, visualize=True, title="Shear Y by 1.25")
#     print(f"shear_y({v}, 1.25) -> {sy}")

#     rl = reflect_line(v, math.pi / 4, visualize=True, title="Reflect Across y=x")
#     print(f"reflect_line({v}, pi/4) -> {rl}")


# def run_projection_demo() -> None:
#     print("\n=== Projection Demo ===")
#     v1 = Vector([3, 2])
#     axis = Vector([1, 0])
#     projected = project_onto(v1, axis)
#     print(f"project_onto({v1}, {axis}) -> {projected}")
#     v1.visualize_transformation(projected, title="Projection onto X-axis")


# def run_eigen_demo() -> None:
#     print("\n=== Eigen Demo ===")
#     matrix = Matrix([[3, 1], [0, 2]])
#     lambdas, vectors = eigen_2x2(matrix, visualize=True)
#     print(f"Matrix:\n{matrix}")
#     print(f"Eigenvalues: {lambdas}")
#     print(f"Eigenvectors: {vectors}")


# def run_grid_demo() -> None:
#     print("\n=== Grid Transformation Demo ===")
#     matrix = Matrix([[1.2, 0.5], [0.2, 1.0]])
#     matrix.visualize_grid_transformation(show=True)


# def main() -> None:
#     run_vector_visualization_demo()
#     run_matrix_factory_demo()
#     run_transform_demo()
#     run_projection_demo()
#     run_eigen_demo()
#     run_grid_demo()


# if __name__ == "__main__":
#     main()

import math
from src.transformations import rotation_matrix, shear_x_matrix, scale_matrix
from src.visualizer import plot_grid_transformation, visualize_eigenvectors
from src.eigen import eigen_2x2
from src.matrix import Matrix

# See what rotation does to space
plot_grid_transformation(rotation_matrix(math.pi / 4))

# See what shear does
plot_grid_transformation(shear_x_matrix(1.5))

# See eigenvectors
A = Matrix([[3, 1], [0, 2]])
lambdas, vecs = eigen_2x2(A)
visualize_eigenvectors(A, vecs, lambdas)
