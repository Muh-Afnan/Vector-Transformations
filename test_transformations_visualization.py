import math
import unittest
from unittest.mock import patch

from src.matrix import Matrix
from src.vector import Vector
from src.transformations import (
    transform,
    rotate_transform,
    reflect_x,
    reflect_y,
    shear_x,
    shear_y,
    reflect_line,
)


class TestTransformationsVisualization(unittest.TestCase):

    def test_transform_without_visualization(self):
        matrix = Matrix([[0, -1], [1, 0]])
        vector = Vector([1, 0])

        result = transform(matrix, vector)

        self.assertAlmostEqual(result.data[0], 0.0, places=6)
        self.assertAlmostEqual(result.data[1], 1.0, places=6)

    @patch("src.visualizer.visualize_transformations")
    def test_transform_with_visualization(self, mock_visualize):
        matrix = Matrix([[0, -1], [1, 0]])
        vector = Vector([1, 0])

        result = transform(matrix, vector, visualize=True, title="Rotate 90")

        mock_visualize.assert_called_once_with([vector], [result], "Rotate 90")

    def test_transform_visualization_requires_2d(self):
        matrix = Matrix([[1, 0, 0], [0, 1, 0]])
        vector = Vector([1, 2, 3])

        with self.assertRaises(ValueError):
            transform(matrix, vector, visualize=True)

    @patch("src.visualizer.visualize_transformations")
    def test_rotate_transform_with_visualization(self, mock_visualize):
        vector = Vector([1, 0])

        result = rotate_transform(vector, math.pi / 2, visualize=True, title="Rotation")

        self.assertAlmostEqual(result.data[0], 0.0, places=6)
        self.assertAlmostEqual(result.data[1], 1.0, places=6)
        mock_visualize.assert_called_once_with([vector], [result], "Rotation")

    @patch("src.visualizer.visualize_transformations")
    def test_reflect_x_with_visualization(self, mock_visualize):
        vector = Vector([2, 3])

        result = reflect_x(vector, visualize=True, title="Reflect X")

        self.assertEqual(result, Vector([2, -3]))
        mock_visualize.assert_called_once_with([vector], [result], "Reflect X")

    @patch("src.visualizer.visualize_transformations")
    def test_reflect_y_with_visualization(self, mock_visualize):
        vector = Vector([2, 3])

        result = reflect_y(vector, visualize=True, title="Reflect Y")

        self.assertEqual(result, Vector([-2, 3]))
        mock_visualize.assert_called_once_with([vector], [result], "Reflect Y")

    @patch("src.visualizer.visualize_transformations")
    def test_shear_x_with_visualization(self, mock_visualize):
        vector = Vector([1, 2])

        result = shear_x(vector, 2, visualize=True, title="Shear X")

        self.assertEqual(result, Vector([5, 2]))
        mock_visualize.assert_called_once_with([vector], [result], "Shear X")

    @patch("src.visualizer.visualize_transformations")
    def test_shear_y_with_visualization(self, mock_visualize):
        vector = Vector([1, 2])

        result = shear_y(vector, 2, visualize=True, title="Shear Y")

        self.assertEqual(result, Vector([1, 4]))
        mock_visualize.assert_called_once_with([vector], [result], "Shear Y")

    @patch("src.visualizer.visualize_transformations")
    def test_reflect_line_with_visualization(self, mock_visualize):
        vector = Vector([0, 1])

        result = reflect_line(vector, 0, visualize=True, title="Reflect Line")

        self.assertEqual(result, Vector([0, -1]))
        mock_visualize.assert_called_once_with([vector], [result], "Reflect Line")


if __name__ == "__main__":
    unittest.main()
