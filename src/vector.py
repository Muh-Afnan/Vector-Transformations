import copy
import math

class Vector:

    def __init__(self, data: list[int | float]):
        """
        Initialise a Vector from a 1D list.
        Raises ValueError if data is empty or not a 1D list.
        """
        if not isinstance(data,list) or not data:
            raise ValueError("Data must be a non-empty list.")

        if any(isinstance(x, list) for x in data):
            raise ValueError("Data must be a 1D list.")

        self.size = len(data)
        self.data = copy.deepcopy(data)  # defensive copy — caller mutation won't affect us
        
    def shape(self) -> tuple[int]:
        """Return (size,)."""
        return (self.size,)
    
    def __repr__(self) -> str:
        return 'Vector([' + ', '.join(str(x) for x in self.data) + '])'
    
    def __eq__(self,other:object)->bool:
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape() != other.shape():
            return False
        return all(
            abs(v1-v2) < 1e-9
            for v1, v2 in zip(self.data, other.data)
        )
    
    def __add__(self, other:object)->"Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape() != other.shape():
            raise ValueError("Vectors must have the same size for addition.")
        result = [a + b for a, b in zip(self.data, other.data)]
        return Vector(result)
    
    def __sub__(self,other:object)->"Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape() != other.shape():
            raise ValueError("Vectors must have the same size for subtraction.")
        result = [a - b for a, b in zip(self.data, other.data)]
        return Vector(result)
        
    def __mul__(self, scalar: int | float) -> "Vector":
        return Vector([x * scalar for x in self.data])

    def __rmul__(self, scalar: int | float) -> "Vector":
        return self.__mul__(scalar)
    
    def dot(self, other: "Vector") -> int | float:
        if self.shape() != other.shape():
            raise ValueError("Vectors must have the same size for dot product.")
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def magnitude(self) -> float:
        return math.sqrt(sum(x ** 2 for x in self.data))
    
    def normalize(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return Vector([x / mag for x in self.data])
    
    def angle_with(self, other: "Vector") -> float:
        if self.shape() != other.shape():
            raise ValueError("Vectors must have the same size to compute angle.")
        dot_prod = self.dot(other)
        mag_self = self.magnitude()
        mag_other = other.magnitude()
        if mag_self == 0 or mag_other == 0:
            raise ValueError("Cannot compute angle with a zero vector.")
        cos_theta = dot_prod / (mag_self * mag_other)
        # Clamp cos_theta to the range [-1, 1] to avoid numerical issues
        cos_theta = max(-1, min(1, cos_theta))
        return math.acos(cos_theta)
    
    def to_matrix(self) -> "object":
        from src.matrix import Matrix  # avoid circular import
        return Matrix([[x] for x in self.data])

    def visualize(
        self,
        color: str = "blue",
        label: str = "Vector",
        xlim: tuple[int | float, int | float] = (-5, 5),
        ylim: tuple[int | float, int | float] = (-5, 5),
        show: bool = True,
    ) -> "object":
        """
        Plot this vector in 2D.
        Raises ValueError if the vector is not 2D.
        """
        if self.size != 2:
            raise ValueError("Visualization is only supported for 2D vectors.")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.arrow(
            0,
            0,
            self.data[0],
            self.data[1],
            head_width=0.1,
            head_length=0.1,
            fc=color,
            ec=color,
        )
        plt.text(self.data[0], self.data[1], label, fontsize=12)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.axhline(0, color="gray", lw=0.5)
        plt.axvline(0, color="gray", lw=0.5)
        plt.grid()
        plt.title(label)

        if show:
            plt.show()

        return plt.gca()

    def visualize_transformation(
        self,
        transformed: "Vector",
        title: str = "Vector Transformation",
    ) -> None:
        """
        Visualize this 2D vector against its transformed 2D vector.
        """
        if self.size != 2 or transformed.size != 2:
            raise ValueError("Transformation visualization is only supported for 2D vectors.")

        from src.visualizer import visualize_transformations

        visualize_transformations([self], [transformed], title)