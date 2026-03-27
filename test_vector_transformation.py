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
    project_onto
)

# =========================
# 🔍 DEBUG RUNNER
# =========================

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


# =========================
# 🧮 VECTOR TESTS
# =========================

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

    def test_equality(self):
        self.assertTrue(Vector([1, 2]) == Vector([1, 2]))
        self.assertTrue(Vector([1.0000000001, 2]) == Vector([1, 2]))

    def test_add(self):
        self.assertEqual(Vector([1, 2]) + Vector([3, 4]), Vector([4, 6]))

    def test_add_invalid(self):
        with self.assertRaises(ValueError):
            Vector([1]) + Vector([1, 2])

    def test_sub(self):
        self.assertEqual(Vector([5, 5]) - Vector([2, 3]), Vector([3, 2]))

    def test_scalar_mul(self):
        self.assertEqual(Vector([1, 2]) * 3, Vector([3, 6]))
        self.assertEqual(3 * Vector([1, 2]), Vector([3, 6]))

    def test_dot(self):
        self.assertEqual(Vector([1, 2]).dot(Vector([3, 4])), 11)

    def test_dot_invalid(self):
        with self.assertRaises(ValueError):
            Vector([1]).dot(Vector([1, 2]))

    def test_dot_commutative(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        self.assertEqual(v1.dot(v2), v2.dot(v1))

    def test_dot_distributive(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        v3 = Vector([5, 6])
        self.assertEqual(v1.dot(v2 + v3), v1.dot(v2) + v1.dot(v3))

    def test_magnitude(self):
        self.assertAlmostEqual(Vector([3, 4]).magnitude(), 5)

    def test_normalize(self):
        v = Vector([3, 4]).normalize()
        self.assertAlmostEqual(v.magnitude(), 1)

    def test_normalize_zero(self):
        with self.assertRaises(ValueError):
            Vector([0, 0]).normalize()

    def test_angle(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        angle = v1.angle_with(v2)
        print(f"\nAngle: {angle}")
        self.assertAlmostEqual(angle, math.pi / 2)

    def test_angle_zero_vector(self):
        with self.assertRaises(ValueError):
            Vector([0, 0]).angle_with(Vector([1, 0]))

    def test_angle_clamping(self):
        v1 = Vector([1, 0])
        v2 = Vector([1e-12, 1])
        angle = v1.angle_with(v2)
        self.assertTrue(0 <= angle <= math.pi)

    def test_to_matrix(self):
        v = Vector([1, 2])
        m = v.to_matrix()
        self.assertEqual(m.data, [[1], [2]])

    @patch("matplotlib.pyplot.show")
    def test_visualize_2d(self, _mock_show):
        axis = Vector([1, 2]).visualize(show=False)
        self.assertIsNotNone(axis)

    def test_visualize_non_2d(self):
        with self.assertRaises(ValueError):
            Vector([1, 2, 3]).visualize(show=False)

    @patch("src.visualizer.visualize_transformations")
    def test_visualize_transformation(self, mock_visualize_transformations):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        v1.visualize_transformation(v2, "Test")
        mock_visualize_transformations.assert_called_once_with([v1], [v2], "Test")

    def test_visualize_transformation_non_2d(self):
        with self.assertRaises(ValueError):
            Vector([1, 2, 3]).visualize_transformation(Vector([1, 2, 3]))


# =========================
# 🧱 MATRIX FACTORIES
# =========================

class TestMatrixFactories(unittest.TestCase):

    def test_rotation_matrix(self):
        m = rotation_matrix(math.pi / 2)
        self.assertAlmostEqual(m.data[0][0], 0, places=6)
        self.assertAlmostEqual(m.data[1][0], 1, places=6)

    def test_scale_matrix(self):
        self.assertEqual(scale_matrix(2, 3).data, [[2, 0], [0, 3]])

    def test_reflect_x_matrix(self):
        self.assertEqual(reflect_x_matrix().data, [[1, 0], [0, -1]])

    def test_reflect_y_matrix(self):
        self.assertEqual(reflect_y_matrix().data, [[-1, 0], [0, 1]])

    def test_reflect_line_matrix(self):
        m = reflect_line_matrix(0)
        self.assertEqual(m.data, [[1, 0], [0, -1]])  # based on your implementation

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
    def test_matrix_visualize_eigenvectors(self, mock_eigen_2x2):
        matrix = Matrix([[2, 0], [0, 1]])
        matrix.visualize_eigenvectors(show=False)
        mock_eigen_2x2.assert_called_once_with(matrix, visualize=False)


# =========================
# 🔄 TRANSFORMATIONS
# =========================

class TestTransformations(unittest.TestCase):

    def test_transform_valid(self):
        v = Vector([1, 0])
        m = rotation_matrix(math.pi / 2)
        result = transform(m, v)

        print(f"\nTransform result: {result}")

        self.assertAlmostEqual(result.data[0], 0, places=6)
        self.assertAlmostEqual(result.data[1], 1, places=6)

    def test_transform_invalid(self):
        with self.assertRaises(ValueError):
            transform(Matrix([[1, 2, 3]]), Vector([1, 2]))

    @patch("src.visualizer.visualize_transformations")
    def test_transform_visualize(self, mock_visualize_transformations):
        vector = Vector([1, 0])
        matrix = rotation_matrix(math.pi / 2)
        result = transform(matrix, vector, visualize=True, title="Rotate")
        self.assertAlmostEqual(result.data[0], 0, places=6)
        self.assertAlmostEqual(result.data[1], 1, places=6)
        mock_visualize_transformations.assert_called_once_with([vector], [result], "Rotate")

    def test_transform_visualize_non_2d(self):
        with self.assertRaises(ValueError):
            transform(Matrix([[1, 0, 0], [0, 1, 0]]), Vector([1, 2, 3]), visualize=True)

    def test_rotate_transform(self):
        v = Vector([1, 0])
        result = rotate_transform(v, math.pi / 2)
        self.assertAlmostEqual(result.data[1], 1, places=6)

    @patch("src.visualizer.visualize_transformations")
    def test_rotate_transform_visualize(self, mock_visualize_transformations):
        v = Vector([1, 0])
        result = rotate_transform(v, math.pi / 2, visualize=True, title="Rotation")
        mock_visualize_transformations.assert_called_once_with([v], [result], "Rotation")

    def test_rotation_inverse(self):
        v = Vector([1, 2])
        theta = math.pi / 4
        rotated = rotate_transform(v, theta)
        back = rotate_transform(rotated, -theta)
        self.assertTrue(v == back)

    def test_rotation_preserves_magnitude(self):
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

    def test_reflect_line(self):
        v = Vector([1, 0])
        result = reflect_line(v, 0)
        self.assertEqual(result, Vector([1, 0]))

    def test_reflect_line(self):
        result = reflect_line(Vector([0, 1]), 0)
        self.assertEqual(result, Vector([0, -1]))


# =========================
# 📐 PROJECTION
# =========================

class TestProjection(unittest.TestCase):

    def test_project(self):
        v1 = Vector([2, 2])
        v2 = Vector([1, 0])
        self.assertEqual(project_onto(v1, v2), Vector([2, 0]))

    def test_project_zero(self):
        with self.assertRaises(ValueError):
            project_onto(Vector([1, 2]), Vector([0, 0]))

    def test_project_idempotent(self):
        v = Vector([3, 4])
        axis = Vector([1, 0])
        once = project_onto(v, axis)
        twice = project_onto(once, axis)
        self.assertEqual(once, twice)

    def test_eigenvector_property(self):
        A = Matrix([[3, 1], [0, 2]])
        lambdas, vectors = eigen_2x2(A)
        for lam, vec in zip(lambdas, vectors):
            Av = transform(A, vec)
            lv = vec * lam
            self.assertEqual(Av, lv)

    def test_scale_preserves_direction(self):
        v = Vector([1, 0])
        result = transform(scale_matrix(3, 3), v)
        self.assertAlmostEqual(v.angle_with(result), 0, places=6)

    def test_double_reflect_x_is_identity(self):
        v = Vector([3, 4])
        self.assertEqual(reflect_x(reflect_x(v)), v)

    def test_project_direction_independent_of_scale(self):
        v = Vector([2, 2])
        p1 = project_onto(v, Vector([1, 0]))
        p2 = project_onto(v, Vector([5, 0]))
        self.assertEqual(p1.normalize(), p2.normalize())


class TestVisualizationIntegration(unittest.TestCase):

    @patch("src.visualizer.visualize_eigenvectors")
    def test_eigen_2x2_visualize(self, mock_visualize_eigenvectors):
        matrix = Matrix([[3, 1], [0, 2]])
        lambdas, vectors = eigen_2x2(matrix, visualize=True)
        self.assertEqual(len(lambdas), 2)
        self.assertEqual(len(vectors), 2)
        mock_visualize_eigenvectors.assert_called_once_with(matrix, vectors)

    @patch("src.visualizer.visualize_transformations")
    def test_plot_grid_transformation(self, mock_visualize_transformations):
        matrix = Matrix([[1, 0], [0, 1]])
        plot_grid_transformation(matrix, show=False)
        self.assertTrue(mock_visualize_transformations.called)


# =========================
# 🚀 RUNNER + SUMMARY
# =========================

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(".")
    runner = DebugTestRunner()
    result = runner.run(suite)

    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Total Run: {result.testsRun}")
    print(f"Failures : {len(result.failures)}")
    print(f"Errors   : {len(result.errors)}")

    if result.failures:
        print("\n❌ Failed Tests:")
        for test, _ in result.failures:
            print(f" - {test}")

    if result.errors:
        print("\n💥 Error Tests:")
        for test, _ in result.errors:
            print(f" - {test}")

    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED")