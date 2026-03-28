"""
visualizer.py

All visualisation functions for Day 2 — Vector Transformations & Eigenvectors.

Three distinct visual types:
    1. visualize_transformation   — single vector before/after a transformation
    2. visualize_eigenvectors     — eigenvectors before/after matrix is applied
    3. plot_grid_transformation   — how a matrix deforms the entire 2D grid

Design decisions:
    - Every function accepts show=True so tests can pass show=False to avoid
      blocking the terminal
    - Dynamic axis limits — never hardcoded, always fit the data
    - Side-by-side subplots for grid transformation so original and result
      are directly comparable
    - Eigenvector plot shows BOTH original and transformed vector so the
      "same direction, different magnitude" property is visually obvious
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.matrix import Matrix
from src.vector import Vector


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _draw_arrow(ax, vector: Vector, color: str, label: str, alpha: float = 1.0) -> None:
    """
    Draw a single 2D vector as an arrow from the origin on the given axes.
    """
    ax.annotate(
        "",
        xy=(vector.data[0], vector.data[1]),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=2,
            alpha=alpha,
        ),
    )
    # Offset the label slightly so it doesn't overlap the arrowhead
    ax.text(
        vector.data[0] * 1.05,
        vector.data[1] * 1.05,
        label,
        color=color,
        fontsize=10,
        fontweight="bold",
    )


def _set_axes(ax, all_vectors: list[Vector], padding: float = 1.0) -> None:
    """
    Set axis limits to fit all vectors with padding.
    Ensures the origin is always visible.
    """
    if not all_vectors:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        return

    all_x = [v.data[0] for v in all_vectors] + [0]
    all_y = [v.data[1] for v in all_vectors] + [0]

    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.8, zorder=0)
    ax.axvline(0, color="gray", lw=0.8, zorder=0)
    ax.grid(True, lw=0.4, alpha=0.6)


# -----------------------------------------------------------------------
# 1. Single transformation visualisation
# -----------------------------------------------------------------------

def visualize_transformation(
    original: Vector,
    transformed: Vector,
    title: str = "Vector Transformation",
    show: bool = True,
) -> plt.Axes:
    """
    Show one vector before (blue) and after (red) a transformation.

    Use this for a single vector. For multiple vectors or a full grid,
    use visualize_multiple_transformations or plot_grid_transformation.

    Example:
        v = Vector([1, 0])
        result = rotate_transform(v, math.pi / 4)
        visualize_transformation(v, result, title="45° Rotation")
    """
    if original.size != 2 or transformed.size != 2:
        raise ValueError("Only 2D vectors can be visualised.")

    fig, ax = plt.subplots(figsize=(7, 7))

    _draw_arrow(ax, original, color="royalblue", label="Original")
    _draw_arrow(ax, transformed, color="crimson", label="Transformed")

    _set_axes(ax, [original, transformed])

    blue_patch = mpatches.Patch(color="royalblue", label="Original")
    red_patch = mpatches.Patch(color="crimson", label="Transformed")
    ax.legend(handles=[blue_patch, red_patch], loc="upper left")
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    if show:
        plt.show()
    return ax


# -----------------------------------------------------------------------
# 2. Eigenvector visualisation
# -----------------------------------------------------------------------

def visualize_eigenvectors(
    matrix: Matrix,
    eigenvectors: list[Vector],
    eigenvalues: list[float],
    show: bool = True,
) -> plt.Axes:
    """
    For each eigenvector, show:
        - The original eigenvector (solid arrow)
        - The transformed eigenvector A @ v (dashed arrow)

    Both arrows point in the same direction — that is the visual proof
    of what an eigenvector is. The only difference is their length (λ).

    The eigenvalue λ is shown in the label so the scaling is clear.

    Example:
        lambdas, vecs = eigen_2x2(A)
        visualize_eigenvectors(A, vecs, lambdas)
    """
    from src.transformations import transform

    if not eigenvectors:
        raise ValueError("No eigenvectors provided.")
    if len(eigenvectors) != len(eigenvalues):
        raise ValueError("eigenvectors and eigenvalues must have the same length.")

    # Colours cycle for each eigenvector pair
    palette = [
        ("royalblue", "steelblue"),
        ("forestgreen", "limegreen"),
        ("darkorange", "gold"),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    all_vectors = []

    for idx, (vec, lam) in enumerate(zip(eigenvectors, eigenvalues)):
        if vec.size != 2:
            raise ValueError("Only 2D eigenvectors can be visualised.")

        solid_color, dash_color = palette[idx % len(palette)]
        transformed = transform(matrix, vec)
        all_vectors.extend([vec, transformed])

        # Original eigenvector — solid
        _draw_arrow(ax, vec, color=solid_color, label=f"v{idx+1}")

        # Transformed eigenvector — shown with alpha to distinguish
        _draw_arrow(
            ax,
            transformed,
            color=dash_color,
            label=f"A·v{idx+1} (λ={lam:.3f})",
            alpha=0.65,
        )

    _set_axes(ax, all_vectors, padding=0.5)

    # Build legend manually
    legend_handles = []
    for idx, (vec, lam) in enumerate(zip(eigenvectors, eigenvalues)):
        solid_color, dash_color = palette[idx % len(palette)]
        legend_handles.append(
            mpatches.Patch(color=solid_color, label=f"v{idx+1} (eigenvector {idx+1})")
        )
        legend_handles.append(
            mpatches.Patch(color=dash_color, label=f"A·v{idx+1}, λ={lam:.3f}")
        )

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(
        f"Eigenvectors of Matrix\n{matrix}\n"
        "(solid = original eigenvector, faded = after transformation)",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    if show:
        plt.show()
    return ax


# -----------------------------------------------------------------------
# 3. Grid transformation
# -----------------------------------------------------------------------

def plot_grid_transformation(
    matrix: Matrix,
    grid_range: int = 5,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Show how a matrix transformation deforms the entire 2D grid.

    Left panel:  original grid (blue)
    Right panel: transformed grid (red)

    Grid lines are drawn as connected line segments — not arrows from the
    origin. This is the correct way to visualise what a matrix does to space.

    The key insight: straight parallel grid lines becoming curved or
    skewed shows exactly what shear, projection, or other transforms do.

    Example:
        plot_grid_transformation(rotation_matrix(math.pi / 4))
        plot_grid_transformation(shear_x_matrix(1.5))
        plot_grid_transformation(scale_matrix(2, 0.5))
    """
    from src.transformations import transform

    coords = range(-grid_range, grid_range + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax_orig, ax_trans = axes

    # --- Draw original grid ---
    # Horizontal lines: fix j, vary i
    for j in coords:
        xs = list(coords)
        ys = [j] * len(xs)
        ax_orig.plot(xs, ys, color="royalblue", lw=0.8, alpha=0.7)

    # Vertical lines: fix i, vary j
    for i in coords:
        xs = [i] * len(list(coords))
        ys = list(coords)
        ax_orig.plot(xs, ys, color="royalblue", lw=0.8, alpha=0.7)

    # --- Draw transformed grid ---
    # Horizontal lines: transform each point, connect them
    all_tx, all_ty = [], []

    for j in coords:
        tx = []
        ty = []
        for i in coords:
            p = transform(matrix, Vector([i, j]))
            tx.append(p.data[0])
            ty.append(p.data[1])
        ax_trans.plot(tx, ty, color="crimson", lw=0.8, alpha=0.7)
        all_tx.extend(tx)
        all_ty.extend(ty)

    # Vertical lines: fix i, vary j
    for i in coords:
        tx = []
        ty = []
        for j in coords:
            p = transform(matrix, Vector([i, j]))
            tx.append(p.data[0])
            ty.append(p.data[1])
        ax_trans.plot(tx, ty, color="crimson", lw=0.8, alpha=0.7)

    # --- Axis formatting ---
    # Original: fixed symmetric limits
    lim = grid_range + 1
    for ax, color in [(ax_orig, "royalblue"), (ax_trans, "crimson")]:
        ax.axhline(0, color="gray", lw=0.8, zorder=0)
        ax.axvline(0, color="gray", lw=0.8, zorder=0)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_aspect("equal")

    ax_orig.set_xlim(-lim, lim)
    ax_orig.set_ylim(-lim, lim)
    ax_orig.set_title("Original Grid", fontsize=12, fontweight="bold", color="royalblue")

    # Transformed: fit to data
    if all_tx and all_ty:
        pad = max(1.0, (max(all_tx) - min(all_tx)) * 0.1)
        ax_trans.set_xlim(min(all_tx) - pad, max(all_tx) + pad)
        ax_trans.set_ylim(min(all_ty) - pad, max(all_ty) + pad)

    ax_trans.set_title("Transformed Grid", fontsize=12, fontweight="bold", color="crimson")

    fig.suptitle(
        f"Grid Transformation\n{matrix}",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    if show:
        plt.show()

    return ax_orig, ax_trans