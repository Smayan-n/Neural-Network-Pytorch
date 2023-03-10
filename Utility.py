import numpy as np
import matplotlib.pyplot as plt


def interpolate(p1, p2, step_size=0.05):
    """yields points along the line connecting p1 and p2"""
    # step size determines the distance between interpolated points

    # calculate the x and y direction between the two points
    direction = (p2[0] - p1[0], p2[1] - p1[1])
    # interpolate the pixel values along the line connecting the two points
    for t in np.arange(0, 1, step_size):
        x = int(round(p1[0] + t * direction[0]))
        y = int(round(p1[1] + t * direction[1]))
        yield x, y


def lerp(a, b, f):
    """return a value between a and b, given a factor f"""
    return a + f * (b - a)


def plot_cost_graph(losses):
    """plots a cost grapgh given losses"""
    plt.plot([i for i in range(len(losses))], losses, color="red", marker="o")

    plt.xlim(0, len(losses))
    plt.ylim(0, 3)

    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.show()
