import numpy as np
from sympy import symbols, Matrix, diff, pprint, expand, lambdify
import matplotlib.pyplot as plt

# Scalars
x1, x2 = symbols("x1 x2")


def gradient_descent(J, x0, name, lims):
        
    # initialize xk = x0
    xk = Matrix([ 
        x0[0], 
        x0[1] 
    ])

    # compute symbolic gradient
    gradJ = Matrix([ 
        diff(J, x1),
        diff(J, x2)
    ])
    pprint(expand(gradJ))
    
    # initialize hyperparameters
    grad_eps = 1e-6
    alpha = 0.08
    kmax = 5000
    
    # for plotting
    x1s = []
    x2s = []
    Js = []
    x1x2 = np.linspace(*lims, 100)
    X1, X2 = np.meshgrid(x1x2, x1x2)
    f_J = lambdify((x1, x2), J, 'numpy')
    f_Js = f_J(X1, X2)


    for k in range(kmax):
        
        # isolate xk components
        x1k = xk[0]
        x2k = xk[1]
        
        # append points for plotting
        x1s.append(float(x1k))
        x2s.append(float(x2k))
        Js.append(float(f_J(x1k, x2k)))
                
        # compute gradient at x = xk
        gradJk = gradJ.subs([ 
            (x1, x1k), 
            (x2, x2k) 
        ])
        
        # compute gradient norm
        gradJk_norm = gradJk.norm()
        
        # check for convergence
        if gradJk_norm <= grad_eps:
            print("k:", k, ", xk:", xk.evalf(3), ", Jk:", J.subs([(x1, x1k), (x2, x2k)]), ", gradJk_norm:", gradJk_norm.evalf(3), "\n\n")
            
            # plotting
            ax = plt.axes(projection="3d")
            ax.plot_surface(X1, X2, f_Js, cmap="viridis", alpha=0.4) # plot J
            ax.plot(x1s, x2s, Js, color="red")  # line
            ax.scatter(x1s, x2s, Js, color="red")  # scatter
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("J")
            ax.set_title(f"J(x1, x2) = {str(J)}\n\nx0 = {x0}\n\nx* = [{float(x1k): .3f}, {float(x2k): .3f}]")
            plt.tight_layout()
            
            # plt.show()
            plt.savefig(f"{name}.pdf")
            plt.close()
            
            return
        
        # check if dk update is necessary
        if k == 0 or (gradJk.T @ dk)[0] >= 0:
            dk = -alpha*gradJk
    
        # update xk
        xk += dk 
        
    print("kmax exceeded\n\n")


if __name__ == "__main__":
    
    # # Question 2a.iii)
    J_a = 3*x1**2 + x2**2
    x0_a = [-5, 5]
    gradient_descent(J_a, x0_a, "q2aiii", x0_a)
    
    # # Question 2b)
    J_b = x1**2 + x2**2
    x0_b = [-5, 5]
    gradient_descent(J_b, x0_b, "q2b", x0_b)

    # # Question 2c.ii.A) / 2c.ii.C)
    J_c = ( 4 - 2.1*x1**2 + (x1**4)/3 ) * x1**2 + x1*x2 + ( -4 + 4*x2**2 ) * x2**2
    x0_cs = [[1.2, 0.2], [-1.5, 0.5], [0.5, -0.5], [0, 0]]
    for x0_c in x0_cs:
        print(x0_c)
        gradient_descent(J_c, x0_c, "q2ciiC_" + "".join(str(x0_c)), [-2, 2])
        
