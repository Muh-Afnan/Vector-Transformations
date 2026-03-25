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
            raise ValueError("Vectors must have the same size for comparison.")
        return all(
            abs(v1-v2) < 1e-9
            for v1, v2 in zip(self.data, other.data)
        )
    
    def _add__(self, other:object)->"Vector":
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
        
    def scale(self, scalar: int | float) -> "Vector":
        result = [scalar * x for x in self.data]
        return Vector(result)
    
    def scale(self, sx:int|float, sy:int|float,*sz:int|float) -> "Vector":
        if self.size == 2:
            
    
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
        return self.scale(1 / mag)  
    
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