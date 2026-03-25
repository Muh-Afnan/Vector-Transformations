import matplotlib.pyplot as plt
from src.matrix import Matrix


def visualizer(vector:Matrix)-> None:
    x,y = vector.data[0][0], vector.data[1][0]

    plt.quiver(0,0,x,y)
    