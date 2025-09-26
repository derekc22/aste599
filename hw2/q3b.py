import numpy as np
from sympy import symbols, Matrix, diff, pprint, expand

# Scalars
x1, x2 = symbols("x1 x2")


def newton_method(J, x0):
    
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
    
    # compute symbolic hessian
    hessJ = Matrix([ 
        [diff(diff(J, x1), x1), diff(diff(J, x1), x2)],
        [diff(diff(J, x2), x1), diff(diff(J, x2), x2)]
    ])
    pprint(expand(hessJ))
        
    # initialize hyperparameters
    grad_eps = 1e-6
    kmax = 5000

    for k in range(kmax):
        
        # isolate xk components
        x1k = xk[0]
        x2k = xk[1]
        
        # compute gradient at x = xk
        gradJk = gradJ.subs([ 
            (x1, x1k), 
            (x2, x2k) 
        ])
        
        # compute gradient norm
        gradJk_norm = gradJk.norm()
        
        # compute hessian at x = xk
        hessJk = hessJ.subs([ 
            (x1, x1k), 
            (x2, x2k) 
        ])
        
        # compute dk
        # dk = -hessJk.inv() @ gradJk 
        dk = -hessJk.LUsolve(gradJk) # does the same thing as above, but more efficiently

        # check for convergence
        if gradJk_norm <= grad_eps:
            print("k:", k, ", Jk:", J.subs([(x1, x1k), (x2, x2k)]), ", xk:", xk.evalf(3), "\n\n")
            return

        # update xk 
        xk += dk
        
    print("kmax exceeded\n\n")


if __name__ == "__main__":
    
    # Question 3c)
    J_c = 3*x1**2 + x2**2
    x0_c = [-50, 50]
    newton_method(J_c, x0_c)
        
