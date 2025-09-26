import numpy as np
from sympy import symbols, lambdify
import matplotlib.pyplot as plt

# Scalars
x, y = symbols("x y")

def main():

    J = (x - 2)**2 + (y - 1)**2
    h1 = x + y - 1.2
    h2 = x**2 + y**2 - 1

    xy = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(xy, xy)

    f_J = lambdify((x, y), J, "numpy")(X, Y)
    f_h1 = lambdify((x, y), h1, "numpy")(X, Y)
    f_h2 = lambdify((x, y), h2, "numpy")(X, Y)
    
    x_sol = (0.9742, 0.2258)

    ax = plt.axes()
    ax.contour(X, Y, f_J, levels=50, cmap="viridis")  # objective function contours
    ax.contour(X, Y, f_h1, levels=[0], colors="k")
    ax.contour(X, Y, f_h2, levels=[0], colors="k")
    plt.suptitle(f"J(x, y) = {str(J)}\n\nh1(x, y) = {str(h1)} = 0, h2(x, y) = {str(h2)} = 0\n\n", fontsize=10)
    ax.scatter(*x_sol, label=f"(x*, y*)={x_sol}", zorder=10, color="b")  # plot solution point
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    
    # plt.show()
    plt.savefig(f"q1i.pdf")
    plt.close()


if __name__ == "__main__":
    main()

