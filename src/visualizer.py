import matplotlib.pyplot as plt
from src.matrix import Matrix
from src.vector import Vector

def plot_vectors(vectors, color, label):
    if not vectors:
        raise ValueError("At least one vector is required for plotting.")
    for v in vectors:
        if v.shape() != (2,):
            raise ValueError("Only 2D vectors are supported for plotting.")

    for index, v in enumerate(vectors):
        plt.arrow(
            0,
            0,
            v.data[0],
            v.data[1],
            head_width=0.1,
            head_length=0.1,
            fc=color,
            ec=color,
            label=label if index == 0 else None,
        )
    plt.text(vectors[-1].data[0], vectors[-1].data[1], label, fontsize=12)

def visualize_transformations(original_vectors, transformed_vectors, title, show: bool = True):
    plt.figure(figsize=(8, 8))
    plot_vectors(original_vectors, 'blue', 'Original')
    plot_vectors(transformed_vectors, 'red', 'Transformed')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid()
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    return plt.gca()

def visualize_eigenvectors(matrix: Matrix, eigenvectors: list[Vector], show: bool = True):
    plt.figure(figsize=(8, 8))
    plot_vectors(eigenvectors, 'green', 'Eigenvectors')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid()
    plt.title(f"Eigenvectors of Matrix:\n{matrix}")
    plt.legend()
    if show:
        plt.show()
    return plt.gca()

def plot_grid_transformation(matrix: Matrix, show: bool = True):
    # Create a grid of points
    x = range(-5, 6)
    y = range(-5, 6)
    original_points = [Vector([i, j]) for i in x for j in y]
    
    # Transform the points
    from src.transformations import transform
    transformed_points = [transform(matrix, p) for p in original_points]
    
    # Visualize the transformation
    return visualize_transformations(
        original_points,
        transformed_points,
        f"Grid Transformation by Matrix:\n{matrix}",
        show=show,
    )
    