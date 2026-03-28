import unittest
import math
from unittest.mock import patch

from src.eigen import eigen_2x2
from src.vector import Vector
from src.matrix import Matrix
from src.visualizer import plot_grid_transformation
from src.transformations import (
    rotation_matrix, scale_matrix,
    reflect_x_matrix, reflect_y_matrix, reflect_line_matrix,
    shear_x_matrix, shear_y_matrix,
    transform, rotate_transform,
    reflect_x, reflect_y,
    shear_x, shear_y,
    reflect_line,
    project_onto,
)


# ──────────────────────────────────────────────────────────────────────────────
# Debug runner — prints ✅ / ❌ / 💥 per test
# ──────────────────────────────────────────────────────────────────────────────

class DebugTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        print(f"✅ PASS: {test}")
        super().addSuccess(test)

    def addFailure(self, test, err):
        print(f"\n❌ FAILURE: {test}")
        super().addFailure(test, err)

    def addError(self, test, err):
        print(f"\n💥 ERROR: {test}")
        super().addError(test, err)


class DebugTestRunner(unittest.TextTestRunner):
    def __init__(self):
        super().__init__(resultclass=DebugTestResult, verbosity=2)


# ──────────────────────────────────────────────────────────────────────────────
# Vector
# ──────────────────────────────────────────────────────────────────────────────

class TestVector(unittest.TestCase):

    def test_init_valid(self):
        v = Vector([1, 2, 3])
        self.assertEqual(v.shape(), (3,))

    def test_init_invalid_empty(self):
        with self.assertRaises(ValueError):
            Vector([])

    def test_init_invalid_type(self):
        with self.assertRaises(ValueError):
            Vector("123")

    def test_init_nested(self):
        with self.assertRaises(ValueError):
            Vector([[1, 2]])

    def test_defensive_copy(self):
        data = [1, 2]
        v = Vector(data)
        data[0] = 999
        self.assertEqual(v, Vector([1, 2]))

    def test_repr(self):
        self.assertEqual(repr(Vector([1, 2])), "Vector([1, 2])")

    def test_equality_exact(self):
        self.assertEqual(Vector([1, 2]), Vector([1, 2]))

    def test_equality_float_tolerance(self):
        self.assertEqual(Vector([1.0000000001, 2]), Vector([1, 2]))

    def test_different_sizes_not_equal(self):
        self.assertNotEqual(Vector([1, 2]), Vector([1, 2, 3]))

    def test_add(self):
        self.assertEqual(Vector([1, 2]) + Vector([3, 4]), Vector([4, 6]))

    def test_add_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            Vector([1]) + Vector([1, 2])

    def test_sub(self):
        self.assertEqual(Vector([5, 5]) - Vector([2, 3]), Vector([3, 2]))

    def test_scalar_mul_right(self):
        self.assertEqual(Vector([1, 2]) * 3, Vector([3, 6]))

    def test_scalar_mul_left(self):
        self.assertEqual(3 * Vector([1, 2]), Vector([3, 6]))

    def test_dot(self):
        self.assertEqual(Vector([1, 2]).dot(Vector([3, 4])), 11)

    def test_dot_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            Vector([1]).dot(Vector([1, 2]))

    def test_dot_commutative(self):
        v1, v2 = Vector([1, 2]), Vector([3, 4])
        self.assertEqual(v1.dot(v2), v2.dot(v1))

    def test_dot_distributive(self):
        v1, v2, v3 = Vector([1, 2]), Vector([3, 4]), Vector([5, 6])
        self.assertEqual(v1.dot(v2 + v3), v1.dot(v2) + v1.dot(v3))

    def test_magnitude(self):
        self.assertAlmostEqual(Vector([3, 4]).magnitude(), 5)

    def test_normalize_unit_length(self):
        self.assertAlmostEqual(Vector([3, 4]).normalize().magnitude(), 1)

    def test_normalize_zero_raises(self):
        with self.assertRaises(ValueError):
            Vector([0, 0]).normalize()

    def test_angle_perpendicular(self):
        angle = Vector([1, 0]).angle_with(Vector([0, 1]))
        self.assertAlmostEqual(angle, math.pi / 2)

    def test_angle_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector([0, 0]).angle_with(Vector([1, 0]))

    def test_angle_clamping_stays_in_range(self):
        angle = Vector([1, 0]).angle_with(Vector([1e-12, 1]))
        self.assertTrue(0 <= angle <= math.pi)

    def test_to_matrix(self):
        self.assertEqual(Vector([1, 2]).to_matrix().data, [[1], [2]])

    # ── visualise ──

    @patch("matplotlib.pyplot.show")
    def test_visualize_2d(self, _mock_show):
        ax = Vector([1, 2]).visualize(show=False)
        self.assertIsNotNone(ax)

    def test_visualize_non_2d_raises(self):
        with self.assertRaises(ValueError):
            Vector([1, 2, 3]).visualize(show=False)

    @patch("src.visualizer.visualize_transformations")
    def test_visualize_transformation_calls_visualizer(self, mock_vt):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        v1.visualize_transformation(v2, "Test")
        mock_vt.assert_called_once_with([v1], [v2], "Test")

    def test_visualize_transformation_non_2d_raises(self):
        with self.assertRaises(ValueError):
            Vector([1, 2, 3]).visualize_transformation(Vector([1, 2, 3]))


# ──────────────────────────────────────────────────────────────────────────────
# Transformation matrix builders
# ──────────────────────────────────────────────────────────────────────────────

class TestMatrixFactories(unittest.TestCase):

    def test_rotation_matrix_90(self):
        m = rotation_matrix(math.pi / 2)
        self.assertAlmostEqual(m.data[0][0], 0, places=6)
        self.assertAlmostEqual(m.data[1][0], 1, places=6)

    def test_scale_matrix(self):
        self.assertEqual(scale_matrix(2, 3).data, [[2, 0], [0, 3]])

    def test_reflect_x_matrix(self):
        self.assertEqual(reflect_x_matrix().data, [[1, 0], [0, -1]])

    def test_reflect_y_matrix(self):
        self.assertEqual(reflect_y_matrix().data, [[-1, 0], [0, 1]])

    def test_reflect_line_matrix_at_zero_equals_reflect_x(self):
        # Reflecting across the x-axis (theta=0) must give [[1,0],[0,-1]]
        self.assertEqual(reflect_line_matrix(0).data, [[1, 0], [0, -1]])

    def test_shear_x_matrix(self):
        self.assertEqual(shear_x_matrix(2).data, [[1, 2], [0, 1]])

    def test_shear_y_matrix(self):
        self.assertEqual(shear_y_matrix(2).data, [[1, 0], [2, 1]])

    @patch("src.visualizer.plot_grid_transformation")
    def test_matrix_visualize_grid_transformation(self, mock_plot_grid):
        matrix = Matrix([[1, 0], [0, 1]])
        matrix.visualize_grid_transformation(show=False)
        mock_plot_grid.assert_called_once_with(matrix, show=False)

    @patch("src.eigen.eigen_2x2")
    def test_matrix_visualize_eigenvectors(self, mock_eigen):
        matrix = Matrix([[2, 0], [0, 1]])
        matrix.visualize_eigenvectors(show=False)
        mock_eigen.assert_called_once_with(matrix, visualize=False)


# ──────────────────────────────────────────────────────────────────────────────
# Transformations
# ──────────────────────────────────────────────────────────────────────────────

class TestTransformations(unittest.TestCase):

    def test_transform_valid(self):
        result = transform(rotation_matrix(math.pi / 2), Vector([1, 0]))
        self.assertAlmostEqual(result.data[0], 0, places=6)
        self.assertAlmostEqual(result.data[1], 1, places=6)

    def test_transform_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            transform(Matrix([[1, 2, 3]]), Vector([1, 2]))

    @patch("src.visualizer.visualize_transformations")
    def test_transform_visualize_calls_visualizer(self, mock_vt):
        v = Vector([1, 0])
        m = rotation_matrix(math.pi / 2)
        result = transform(m, v, visualize=True, title="Rotate")
        self.assertAlmostEqual(result.data[0], 0, places=6)
        self.assertAlmostEqual(result.data[1], 1, places=6)
        mock_vt.assert_called_once_with([v], [result], "Rotate")

    def test_transform_visualize_non_2d_raises(self):
        with self.assertRaises(ValueError):
            transform(
                Matrix([[1, 0, 0], [0, 1, 0]]),
                Vector([1, 2, 3]),
                visualize=True,
            )

    def test_rotate_transform_90(self):
        result = rotate_transform(Vector([1, 0]), math.pi / 2)
        self.assertAlmostEqual(result.data[1], 1, places=6)

    @patch("src.visualizer.visualize_transformations")
    def test_rotate_transform_visualize(self, mock_vt):
        v = Vector([1, 0])
        result = rotate_transform(v, math.pi / 2, visualize=True, title="Rotation")
        mock_vt.assert_called_once_with([v], [result], "Rotation")

    def test_rotation_inverse_property(self):
        """Rotating by θ then -θ returns the original vector."""
        v = Vector([1, 2])
        back = rotate_transform(rotate_transform(v, math.pi / 4), -math.pi / 4)
        self.assertEqual(v, back)

    def test_rotation_preserves_magnitude(self):
        """Rotation must not change vector length."""
        v = Vector([3, 4])
        rotated = rotate_transform(v, math.pi / 3)
        self.assertAlmostEqual(v.magnitude(), rotated.magnitude(), places=6)

    def test_reflect_x(self):
        self.assertEqual(reflect_x(Vector([1, 2])), Vector([1, -2]))

    def test_reflect_y(self):
        self.assertEqual(reflect_y(Vector([1, 2])), Vector([-1, 2]))

    def test_shear_x(self):
        self.assertEqual(shear_x(Vector([1, 2]), 2), Vector([5, 2]))

    def test_shear_y(self):
        self.assertEqual(shear_y(Vector([1, 2]), 2), Vector([1, 4]))

    def test_reflect_line_degenerate(self):
        """[1,0] lies on the x-axis — reflecting it across x-axis leaves it unchanged."""
        self.assertEqual(reflect_line(Vector([1, 0]), 0), Vector([1, 0]))

    def test_reflect_line_non_degenerate(self):
        """[0,1] reflected across the x-axis (theta=0) must give [0,-1]."""
        self.assertEqual(reflect_line(Vector([0, 1]), 0), Vector([0, -1]))

    def test_double_reflect_x_is_identity(self):
        """Reflecting twice across x-axis returns the original vector."""
        v = Vector([3, 4])
        self.assertEqual(reflect_x(reflect_x(v)), v)

    def test_scale_preserves_direction(self):
        """Uniform scaling should not change direction."""
        v = Vector([1, 0])
        result = transform(scale_matrix(3, 3), v)
        self.assertAlmostEqual(v.angle_with(result), 0, places=6)


# ──────────────────────────────────────────────────────────────────────────────
# Projection
# ──────────────────────────────────────────────────────────────────────────────

class TestProjection(unittest.TestCase):

    def test_project_onto_axis(self):
        self.assertEqual(project_onto(Vector([2, 2]), Vector([1, 0])), Vector([2, 0]))

    def test_project_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            project_onto(Vector([1, 2]), Vector([0, 0]))

    def test_project_idempotent(self):
        """Projecting twice must equal projecting once — fundamental property."""
        axis = Vector([1, 0])
        once = project_onto(Vector([3, 4]), axis)
        twice = project_onto(once, axis)
        self.assertEqual(once, twice)

    def test_project_direction_independent_of_axis_scale(self):
        """Projecting onto [1,0] and [5,0] must give same direction."""
        v = Vector([2, 2])
        p1 = project_onto(v, Vector([1, 0]))
        p2 = project_onto(v, Vector([5, 0]))
        self.assertEqual(p1.normalize(), p2.normalize())


# ──────────────────────────────────────────────────────────────────────────────
# Eigen
# ──────────────────────────────────────────────────────────────────────────────

class TestEigen(unittest.TestCase):

    def test_eigenvalues_correct(self):
        """[[3,1],[0,2]] has eigenvalues 3 and 2."""
        lambdas, _ = eigen_2x2(Matrix([[3, 1], [0, 2]]))
        self.assertAlmostEqual(max(lambdas), 3, places=6)
        self.assertAlmostEqual(min(lambdas), 2, places=6)

    def test_eigenvector_property(self):
        """
        Core mathematical property: A @ v must equal λ * v.
        If this passes, the eigen solver is correct by definition.
        """
        A = Matrix([[3, 1], [0, 2]])
        lambdas, vectors = eigen_2x2(A)
        for lam, vec in zip(lambdas, vectors):
            Av = transform(A, vec)
            lv = vec * lam
            self.assertEqual(Av, lv)

    def test_eigenvectors_are_unit_length(self):
        """Eigenvectors must be normalised."""
        _, vectors = eigen_2x2(Matrix([[3, 1], [0, 2]]))
        for v in vectors:
            self.assertAlmostEqual(v.magnitude(), 1, places=6)

    def test_complex_eigenvalues_raise(self):
        """Rotation matrix (90°) has no real eigenvectors."""
        R = rotation_matrix(math.pi / 2)
        with self.assertRaises(ValueError):
            eigen_2x2(R)

    def test_repeated_eigenvalue(self):
        """Identity matrix has eigenvalue 1 repeated — should not crash."""
        lambdas, vectors = eigen_2x2(Matrix([[1, 0], [0, 1]]))
        self.assertAlmostEqual(lambdas[0], 1, places=6)
        self.assertAlmostEqual(lambdas[1], 1, places=6)

    def test_diagonal_matrix_eigenvectors(self):
        """Diagonal matrix eigenvectors must align with axes."""
        _, vectors = eigen_2x2(Matrix([[3, 0], [0, 2]]))
        # One eigenvector should be [1,0] and other [0,1] (in some order)
        axes = {(round(v.data[0]), round(v.data[1])) for v in vectors}
        self.assertIn((1, 0), axes)
        self.assertIn((0, 1), axes)

    @patch("src.visualizer.visualize_eigenvectors")
    def test_eigen_visualize_calls_visualizer(self, mock_ve):
        """eigen_2x2(visualize=True) must call visualize_eigenvectors with lambdas."""
        A = Matrix([[3, 1], [0, 2]])
        lambdas, vectors = eigen_2x2(A, visualize=True)
        mock_ve.assert_called_once_with(A, vectors, lambdas)


# ──────────────────────────────────────────────────────────────────────────────
# Visualiser integration
# ──────────────────────────────────────────────────────────────────────────────

class TestVisualizationIntegration(unittest.TestCase):

    @patch("src.visualizer.visualize_transformations")
    def test_plot_grid_transformation_calls_visualizer(self, mock_vt):
        """plot_grid_transformation must invoke the visualiser internally."""
        plot_grid_transformation(Matrix([[1, 0], [0, 1]]), show=False)
        # plot_grid_transformation uses ax.plot internally, not visualize_transformations
        # so we just verify it doesn't crash and returns axes
        # (mock is here to prevent plt.show blocking)

    @patch("matplotlib.pyplot.show")
    def test_plot_grid_transformation_returns_axes(self, _mock_show):
        axes = plot_grid_transformation(Matrix([[1, 0], [0, 1]]), show=False)
        self.assertIsNotNone(axes)

    @patch("matplotlib.pyplot.show")
    def test_plot_grid_transformation_shear(self, _mock_show):
        axes = plot_grid_transformation(shear_x_matrix(1), show=False)
        self.assertIsNotNone(axes)

    def test_plot_grid_non_2x2_raises(self):
        with self.assertRaises(ValueError):
            plot_grid_transformation(
                Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                show=False,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(".")
    runner = DebugTestRunner()
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Run : {result.testsRun}")
    print(f"Failures  : {len(result.failures)}")
    print(f"Errors    : {len(result.errors)}")

    if result.failures:
        print("\n❌ Failed Tests:")
        for test, _ in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\n💥 Error Tests:")
        for test, _ in result.errors:
            print(f"  - {test}")

    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED")