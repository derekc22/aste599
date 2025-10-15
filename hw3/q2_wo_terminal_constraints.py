import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os
# np.random.seed(12)



def q2b():
        
    # geometric
    L = 1
    
    # discretize
    dt = 0.5
    N = 40
    t_max = N * dt

    # define state, control dim
    nv = 4
    nu = 2
    
    # define δ, a control constraints
    U_lim = [0.7, 0.2]
    
    # define values for x0, xf
    x0_val = np.array([0, 0, 0, 0])
    xf_val = np.array([5, 3, 0, 0])

    # define cost matrices
    Q = ca.DM(np.eye(nv))
    R = ca.DM(np.eye(nu))
    H = ca.DM(np.eye(nv))

    # define dynamics (forward/explicit euler integration)
    def f(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return ca.vcat([dx, dy, dtheta, dv])



    # initialize optimizer
    opti = ca.Opti()

    # define state, control
    X = opti.variable(nv, N+1)
    U = opti.variable(nu, N)

    # declare, set x0
    x0 = opti.parameter(nv)
    opti.set_value(x0, x0_val)

    # declare, set xf
    xf = opti.parameter(nv)
    opti.set_value(xf, xf_val)

    

    # build objective function
    J = 0
    for k in range(N):
        # running cost
        xk = X[:, k]
        uk = U[:, k]
        J += ca.mtimes([(xk-xf).T, Q, (xk-xf)]) + ca.mtimes([uk.T, R, uk])
        
        # dynamics constraint
        xk_next = X[:, k+1]
        opti.subject_to(xk_next == xk + dt * f(xk, uk))

    # terminal cost
    xN = X[:, N]
    J += ca.mtimes([(xN-xf).T, H, (xN-xf)])



    # set constraints
    opti.subject_to(X[:, 0] == x0) # initial constraint, x0
    opti.subject_to(opti.bounded(-U_lim[0], U[0, :], U_lim[0])) # control constraint, δ
    opti.subject_to(opti.bounded(-U_lim[1], U[1, :], U_lim[1])) # control constraint, a
    


    # set objective
    opti.minimize(J)

    # set solver
    opti.solver("ipopt")
    
    # solve
    sol = opti.solve()

    # extract solution
    X_sol = sol.value(X)
    U_sol = sol.value(U)
    Jstar = sol.value(J)
    
    # plot
    t = np.linspace(0, t_max, N+1)
    plt.figure(figsize=(9, 11))
    plt.subplot(5,1,1)
    plt.plot(X_sol[0, :], X_sol[1, :], label='trajectory')
    plt.plot(x0_val[0], x0_val[1], 'go', label='start')
    plt.plot(xf_val[0], xf_val[1], 'ro', label='goal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()

    plt.subplot(5,1,2)
    plt.plot(t, X_sol[2, :])
    plt.xlabel('t [s]')
    plt.ylabel('θ [rad]')
    plt.grid()
    
    plt.subplot(5,1,3)
    plt.plot(t, X_sol[3, :])
    plt.ylabel('v [ms⁻¹]')
    plt.xlabel('t [s]')
    plt.grid()
    
    plt.subplot(5,1,4)
    plt.step(t[:-1], U_sol[0, :])
    plt.ylabel('δ [rad]')
    plt.xlabel('t [s]')
    plt.grid()
    
    plt.subplot(5,1,5)
    plt.step(t[:-1], U_sol[1, :])
    plt.ylabel('a [ms⁻²]')
    plt.xlabel('t [s]')
    plt.grid()
    
    plt.suptitle(f'optimal cost, J* = {Jstar:.2f}')
    plt.tight_layout()
    plt.savefig("plots/q2b.pdf")
    plt.close()



def q2c():
        
    # geometric
    L = 1
    
    # define state, control dim
    nv = 4
    nu = 2
    
    # define δ, a control constraints
    U_lim = [0.7, 0.2]
    
    # define values for x0, xf
    x0_val = np.array([0, 0, 0, 0])
    xf_val = np.array([5, 3, 0, 0])

    # define cost matrices
    Q = ca.DM(np.eye(nv))
    R = ca.DM(np.eye(nu))
    H = ca.DM(np.eye(nv))

    # define dynamics (forward/explicit euler integration)
    def f(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return ca.vcat([dx, dy, dtheta, dv])



    # discretize
    # variable horizon lengths, same total time
    dts = [2, 0.2]
    Ns = [10, 100]
    
    for dt, N in zip(dts, Ns):
        
        t_max = N * dt
        
        # initialize optimizer
        opti = ca.Opti()

        # define state, control
        X = opti.variable(nv, N+1)
        U = opti.variable(nu, N)

        # declare, set x0
        x0 = opti.parameter(nv)
        opti.set_value(x0, x0_val)

        # declare, set xf
        xf = opti.parameter(nv)
        opti.set_value(xf, xf_val)

        

        # build objective function
        J = 0
        for k in range(N):
            # running cost
            xk = X[:, k]
            uk = U[:, k]
            J += ca.mtimes([(xk-xf).T, Q, (xk-xf)]) + ca.mtimes([uk.T, R, uk])
            
            # dynamics constraint
            xk_next = X[:, k+1]
            opti.subject_to(xk_next == xk + dt * f(xk, uk))

        # terminal cost
        xN = X[:, N]
        J += ca.mtimes([(xN-xf).T, H, (xN-xf)])



        # set constraints
        opti.subject_to(X[:, 0] == x0) # initial constraint, x0
        opti.subject_to(opti.bounded(-U_lim[0], U[0, :], U_lim[0])) # control constraint, δ
        opti.subject_to(opti.bounded(-U_lim[1], U[1, :], U_lim[1])) # control constraint, a
        
        # set new crater constraint
        opti.subject_to((X[0, :] - 2)**2 + (X[1, :] - 2)**2 >= 1.0)



        # set objective
        opti.minimize(J)

        # set solver
        opti.solver("ipopt")
        
        # solve
        sol = opti.solve()

        # extract solution
        X_sol = sol.value(X)
        U_sol = sol.value(U)
        Jstar = sol.value(J)
        
        # plot
        t = np.linspace(0, t_max, N+1)
        plt.figure(figsize=(9, 11))
        plt.subplot(5,1,1)
        plt.plot(X_sol[0, :], X_sol[1, :], label='trajectory')
        plt.plot(x0_val[0], x0_val[1], 'go', label='start')
        plt.plot(xf_val[0], xf_val[1], 'ro', label='goal')
        # plot crater
        theta = np.linspace(0, 2*np.pi, 100)
        crater_x = 2 + 1.0 * np.cos(theta)
        crater_y = 2 + 1.0 * np.sin(theta)
        plt.plot(crater_x, crater_y, 'k--', label='crater')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        plt.legend()

        plt.subplot(5,1,2)
        plt.plot(t, X_sol[2, :])
        plt.xlabel('t [s]')
        plt.ylabel('θ [rad]')
        plt.grid()
        
        plt.subplot(5,1,3)
        plt.plot(t, X_sol[3, :])
        plt.ylabel('v [ms⁻¹]')
        plt.xlabel('t [s]')
        plt.grid()
        
        plt.subplot(5,1,4)
        plt.step(t[:-1], U_sol[0, :])
        plt.ylabel('δ [rad]')
        plt.xlabel('t [s]')
        plt.grid()
        
        plt.subplot(5,1,5)
        plt.step(t[:-1], U_sol[1, :])
        plt.ylabel('a [ms⁻²]')
        plt.xlabel('t [s]')
        plt.grid()
        
        plt.suptitle(f'optimal cost J* = {Jstar:.2f}, dt={dt}, N={N}')
        plt.tight_layout()
        plt.savefig(f"plots/q2c_dt{dt}_N{N}.pdf")
        plt.close()



def q2e():
        
    # geometric
    L = 1
    
    # discretize
    dt = 0.5
    N = 40
    t_max = N * dt

    # define state, control dim
    nv = 4
    nu = 2
    
    # define δ, a control constraints
    U_lim = [0.7, 0.2]
    
    # define values for x0, xf
    x0_val = np.array([0, 0, 0, 0])
    xf_val = np.array([5, 3, 0, 0])

    # define cost matrices
    Q = ca.DM(np.eye(nv))
    R = ca.DM(np.eye(nu))
    H = ca.DM(np.eye(nv))

    # define dynamics (forward/explicit euler integration)
    def f(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return ca.vcat([dx, dy, dtheta, dv])



    # initialize optimizer
    opti = ca.Opti()

    # define state, control
    X = opti.variable(nv, N+1)
    U = opti.variable(nu, N)

    # declare, set x0
    x0 = opti.parameter(nv)
    opti.set_value(x0, x0_val)

    # declare, set xf
    xf = opti.parameter(nv)
    opti.set_value(xf, xf_val)

    

    # build objective function
    J = 0
    for k in range(N):
        # running cost
        xk = X[:, k]
        uk = U[:, k]
        J += ca.mtimes([(xk-xf).T, Q, (xk-xf)]) + ca.mtimes([uk.T, R, uk])
        
        # dynamics constraint
        xk_next = X[:, k+1]
        opti.subject_to(xk_next == xk + dt * f(xk, uk))

    # terminal cost
    xN = X[:, N]
    J += ca.mtimes([(xN-xf).T, H, (xN-xf)])



    # set constraints
    opti.subject_to(X[:, 0] == x0) # initial constraint, x0
    opti.subject_to(opti.bounded(-U_lim[0], U[0, :], U_lim[0])) # control constraint, δ
    opti.subject_to(opti.bounded(-U_lim[1], U[1, :], U_lim[1])) # control constraint, a

    # set new crater constraint
    opti.subject_to((X[0, :] - 2)**2 + (X[1, :] - 2)**2 >= 1.0)



    # set objective
    opti.minimize(J)

    # set solver
    opti.solver("ipopt")
    
    # solve
    sol = opti.solve()

    # extract solution
    X_sol = sol.value(X)
    U_sol = sol.value(U)
    Jstar = sol.value(J)
    


    ## now, evolve the system forward in time by applying U_sol in an open-loop manner while also applying a disturbance
    
    # define non-CasADi (i.e. pure NumPy) dynamics (forward/explicit euler integration)
    def f_np(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return np.array([dx, dy, dtheta, dv])
    


    # pre-sample N Gaussian disturbances
    w = np.random.multivariate_normal(np.zeros(nv), np.diag([0.01]*nv), N) # shape (N, nv)



    # store the open-loop trajectory for plotting
    x_ol = np.zeros((N+1, nv))
    x_ol[0] = x0_val

    xk = x0_val
    
    for k in range(N):
        uk = U_sol[:, k]
        
        # apply uk open-loop with disturbance wk
        xk_1 = xk + dt * f_np(xk, uk) + w[k, :]
        
        # add to trajectory
        x_ol[k+1] = xk_1
        
        # update current state
        xk = xk_1
        

    
    # plot
    t = np.linspace(0, t_max, N+1)
    plt.figure(figsize=(9, 11))
    plt.subplot(5,1,1)
    plt.plot(X_sol[0, :], X_sol[1, :], label='det. dynamics')
    plt.plot(x_ol[:, 0], x_ol[:, 1], label='stoch. dynamics')
    plt.plot(x0_val[0], x0_val[1], 'go', label='start')
    plt.plot(xf_val[0], xf_val[1], 'ro', label='goal')
    # plot crater
    theta = np.linspace(0, 2*np.pi, 100)
    crater_x = 2 + 1.0 * np.cos(theta)
    crater_y = 2 + 1.0 * np.sin(theta)
    plt.plot(crater_x, crater_y, 'k--', label='crater')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()

    plt.subplot(5,1,2)
    plt.plot(t, X_sol[2, :], label='det. dynamics')
    plt.plot(t, x_ol[:, 2], label='stoch. dynamics')
    plt.xlabel('t [s]')
    plt.ylabel('θ [rad]')
    plt.grid()
    plt.legend()
    
    plt.subplot(5,1,3)
    plt.plot(t, X_sol[3, :], label='det. dynamics')
    plt.plot(t, x_ol[:, 3], label='stoch. dynamics')
    plt.ylabel('v [ms⁻¹]')
    plt.xlabel('t [s]')
    plt.grid()
    plt.legend()
    
    plt.subplot(5,1,4)
    plt.step(t[:-1], U_sol[0, :])
    plt.ylabel('δ [rad]')
    plt.xlabel('t [s]')
    plt.grid()
    
    plt.subplot(5,1,5)
    plt.step(t[:-1], U_sol[1, :])
    plt.ylabel('a [ms⁻²]')
    plt.xlabel('t [s]')
    plt.grid()
    
    plt.suptitle(f'optimal cost, J* = {Jstar:.2f}')
    plt.tight_layout()
    plt.savefig("plots/q2e.pdf")
    plt.close()



def q2f():
        
    # geometric
    L = 1
    
    # discretize
    dt = 0.5
    N = 40
    t_max = N * dt

    # define state, control dim
    nv = 4
    nu = 2
    
    # define δ, a control constraints
    U_lim = [0.7, 0.2]
    
    # define values for initial x0, xf
    x0_val = np.array([0, 0, 0, 0])
    xf_val = np.array([5, 3, 0, 0])

    # define cost matrices
    Q = ca.DM(np.eye(nv))
    R = ca.DM(np.eye(nu))
    H = ca.DM(np.eye(nv))

    # define dynamics (forward/explicit euler integration)
    def f(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return ca.vcat([dx, dy, dtheta, dv])



    # pre-sample N Gaussian disturbances
    disturbances = ["low", "high"]
    w_low = np.random.multivariate_normal(np.zeros(nv), np.diag([0.001]*nv), N) # shape (N, nv)
    w_high = np.random.multivariate_normal(np.zeros(nv), np.diag([0.1]*nv), N) # shape (N, nv)



    for d in disturbances:

        # initialize optimizer
        opti = ca.Opti()

        # define state, control
        X = opti.variable(nv, N+1)
        U = opti.variable(nu, N)
        
        # declare x0
        x0 = opti.parameter(nv)
        
        # declare, set xf
        xf = opti.parameter(nv)
        opti.set_value(xf, xf_val)
        


        # build objective function
        J = 0
        for k in range(N):
            # running cost
            xk = X[:, k]
            uk = U[:, k]
            J += ca.mtimes([(xk-xf).T, Q, (xk-xf)]) + ca.mtimes([uk.T, R, uk])
            
            # dynamics constraint
            xk_next = X[:, k+1]
            opti.subject_to(xk_next == xk + dt * f(xk, uk))

        # terminal cost
        xN = X[:, N]
        J += ca.mtimes([(xN-xf).T, H, (xN-xf)])



        # set constraints
        opti.subject_to(X[:, 0] == x0) # initial constraint, x0
        opti.subject_to(opti.bounded(-U_lim[0], U[0, :], U_lim[0])) # control constraint, δ
        opti.subject_to(opti.bounded(-U_lim[1], U[1, :], U_lim[1])) # control constraint, a
        
        # set new crater constraint
        opti.subject_to((X[0, :] - 2)**2 + (X[1, :] - 2)**2 >= 1.0)



        # set objective
        opti.minimize(J)

        # set solver
        opti.solver("ipopt")



        # define function to solve the above OCP at variable starting points, x0
        def solve(x0_val):

            # set x0
            opti.set_value(x0, x0_val)
            
            # solve
            sol = opti.solve()

            # extract solution
            U_sol = sol.value(U)
            
            return U_sol
        


        ## now, evolve the system forward in time by re-computing and applying U_sol at each time in an closed-loop manner step while also applying a disturbance
        
        # define non-CasADi (i.e. pure NumPy) dynamics (forward/explicit euler integration)
        def f_np(x, u):
            theta, v = x[2], x[3]
            delta, a = u[0], u[1]
            
            dx = v * np.cos(theta)
            dy = v * np.sin(theta)
            dtheta = (v / L) * delta
            dv = a
            return np.array([dx, dy, dtheta, dv])


        # store the closed-loop trajectory for plotting
        x_cl = np.zeros((N+1, nv))
        x_cl[0] = x0_val
        
        # store the closed-loop control for plotting
        u_cl = np.zeros((N, nu))

        xk = x0_val
        
        if d == "low":
            w = w_low
        elif d == "high":
            w = w_high
        
        for k in range(N):
            
            # re-solve at each step
            U_sol = solve(np.array(xk))
            
            # take the first control in the computed solution
            uk = U_sol[:, 0]
            
            # apply uk closed-loop with disturbance wk
            xk_1 = xk + dt * f_np(xk, uk) + w[k, :]
            
            # add xk+1 to trajectory
            x_cl[k+1] = xk_1
            # add uk to control trajectory
            u_cl[k] = uk
            
            # update current state
            xk = xk_1
        


        # plot
        t = np.linspace(0, t_max, N+1)
        plt.figure(figsize=(9, 11))
        plt.subplot(5,1,1)
        plt.plot(x_cl[:, 0], x_cl[:, 1], label='trajectory')
        plt.plot(x0_val[0], x0_val[1], 'go', label='start')
        plt.plot(xf_val[0], xf_val[1], 'ro', label='goal')
        # plot crater
        theta = np.linspace(0, 2*np.pi, 100)
        crater_x = 2 + 1.0 * np.cos(theta)
        crater_y = 2 + 1.0 * np.sin(theta)
        plt.plot(crater_x, crater_y, 'k--', label='crater')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        plt.legend()

        plt.subplot(5,1,2)
        plt.plot(t, x_cl[:, 2])
        plt.xlabel('t [s]')
        plt.ylabel('θ [rad]')
        plt.grid()
        
        plt.subplot(5,1,3)
        plt.plot(t, x_cl[:, 3])
        plt.ylabel('v [ms⁻¹]')
        plt.xlabel('t [s]')
        plt.grid()
        
        plt.subplot(5,1,4)
        plt.step(t[:-1], u_cl[:, 0])
        plt.ylabel('δ [rad]')
        plt.xlabel('t [s]')
        plt.grid()
        
        plt.subplot(5,1,5)
        plt.step(t[:-1], u_cl[:, 1])
        plt.ylabel('a [ms⁻²]')
        plt.xlabel('t [s]')
        plt.grid()

        plt.suptitle(f'stoch. dynamics, {d} wk')
        plt.tight_layout()
        plt.savefig(f"plots/q2f_{d}.pdf")
        plt.close()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    q2b()
    q2c()
    q2e()
    q2f()
