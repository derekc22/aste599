import numpy as np
from sympy import symbols, Matrix, diff, pprint, expand, lambdify
import matplotlib.pyplot as plt

# Scalars
x, y = symbols("x y")

def plot_grads(x_sol, y_sol):

    J = (x - 2)**2 + (y - 1)**2
    h1 = x + y - 1.2
    h2 = x**2 + y**2 - 1
    
    gradJ = Matrix([ 
        diff(J, x),
        diff(J, y)
    ])
    gradh1 = Matrix([ 
        diff(h1, x),
        diff(h1, y)
    ])
    gradh2 = Matrix([ 
        diff(h2, x),
        diff(h2, y)
    ])
    
    gradJ_f = lambdify((x, y), gradJ, "numpy")
    gradh1_f = lambdify((x, y), gradh1, "numpy")
    gradh2_f = lambdify((x, y), gradh2, "numpy")

    gradJ_f_sol = gradJ_f(x_sol, y_sol)
    gradh1_f_sol = gradh1_f(x_sol, y_sol)
    gradh2_f_sol = gradh2_f(x_sol, y_sol)
        
    ax = plt.axes()

    xs = np.linspace(x_sol - 0.5, x_sol + 0.5, 5)
    ys = np.linspace(y_sol - 0.5, y_sol + 0.5, 5)
    for xi in xs:
        for yi in ys:
            ax.quiver(xi, yi, gradJ_f(xi, yi)[0], gradJ_f(xi, yi)[1], angles="xy", scale_units="xy", scale=8, color="r", alpha=0.5)

    ax.quiver(x_sol, y_sol, gradJ_f_sol[0], gradJ_f_sol[1], angles="xy", scale_units="xy", scale=8, color="r", label="∇J")
    
    ax.quiver(x_sol, y_sol, gradh1_f_sol[0], gradh1_f_sol[1], angles="xy", scale_units="xy", scale=8, color="b", label="∇h1")
    ax.quiver(x_sol, y_sol, gradh2_f_sol[0], gradh2_f_sol[1], angles="xy", scale_units="xy", scale=8, color="g", label="∇h2")
    
    ax.scatter(x_sol, y_sol, label=f"(x*, y*)={x_sol}, {y_sol}", zorder=10, color="b")  # plot solution point    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid()
    
    # plt.suptitle(f"J(x, y) = {str(J)}\n\nh1(x, y) = {str(h1)} = 0\n\n", fontsize=10)
    plt.suptitle(f"J(x, y) = {str(J)}\n\nh1(x, y) = {str(h1)} = 0, h2(x, y) = {str(h2)} = 0\n\n", fontsize=10)

    # plt.show()
    plt.savefig(f"q1j_[{x_sol}, {y_sol}].pdf")
    plt.close()


if __name__ == "__main__":
    # sol_h1 = [1.1, 0.1]
    # plot_grads(*sol_h1)
    
    sol_h1h2 = [0.9742, 0.2258]
    plot_grads(*sol_h1h2)

