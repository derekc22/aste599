import numpy as np
import casadi as ca
from plot import plot

def dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, sigma, mode):
    
    assert mode in ("gauss-seidel", "jacobi"), f"Invalid mode: {mode}"
    
    t_max = N * dt
    
    # disturbances, per agent
    w = [np.random.multivariate_normal(np.zeros(nx), np.diag([sigma] * nx), N) for _ in range(Z)]

    # cost matrices
    Q = ca.DM(np.eye(nx))
    R = ca.DM(np.eye(nu))
    H = ca.DM(np.eye(nx))

    pred_X = np.zeros((Z, nx, N + 1))
    pred_U = np.zeros((Z, nu, N))

    # build a local OCP for one agent, with other agents' XY as parameters
    def build_agent_opti(z):
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx, 1)
        xf = opti.parameter(nx, 1)

        # set final state
        opti.set_value(xf, xf_val[z, :].reshape((nx, 1)))
        
        # control bounds and initial condition constraint
        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(-U_lim[0], U[0, :], U_lim[0]))
        opti.subject_to(opti.bounded(-U_lim[1], U[1, :], U_lim[1]))

        # obstacle constraint, center (2,2), radius 1
        opti.subject_to((X[0, :] - 2) ** 2 + (X[1, :] - 2) ** 2 >= 1.0)
        
        # other agents' predicted positions over horizon
        XY_others = [opti.parameter(2, N + 1) for _ in range(Z - 1)]
        
        # build objective function
        J = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            J += ca.mtimes([(xk - xf).T, Q, (xk - xf)]) + ca.mtimes([uk.T, R, uk])

            # forward Euler
            x_next = xk + dt * f(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)

            # collision avoidance with other agents' broadcast predictions
            for XY_z in XY_others:
                opti.subject_to(ca.sumsqr(X[0:2, k] - XY_z[:, k]) >= d_min ** 2)

        # terminal cost
        xN = X[:, N]
        J += ca.mtimes([(xN - xf).T, H, (xN - xf)])
        
        # push initial interpolated predictions for warm-starting
        x0_z = x0_val[z, :].reshape(nx, 1)
        xf_z = xf_val[z, :].reshape(nx, 1)
        pred_X[z] = np.hstack([x0_z + (k / float(N)) * (xf_z - x0_z) for k in range(N + 1)])

        opti.minimize(J)
        opti.solver("ipopt")
        return {"opti": opti, "X": X, "U": U, "x0": x0, "xf": xf, "XY_others": XY_others, "J" : J}

    # build agents and set goals
    agents = [build_agent_opti(z) for z in range(Z)]
    
    def shift_pred(X):
        return np.hstack([X[:, 1:], X[:, -1:]])

    # helpers
    def set_XY_others(z):
        i = 0
        for j in range(Z):
            if j == z:
                continue
            agents[z]["opti"].set_value(agents[z]["XY_others"][i], pred_X[j][0:2, :])
            i += 1

    # logs for plotting
    x_cl = np.zeros((Z, nx, N + 1), dtype=float)
    x_cl[:, :, 0] = x0_val.copy()
    u_cl = np.zeros((Z, nu, N), dtype=float)
    J_cl = np.zeros((Z, N))

    Xk = x0_val.copy()

    # receding-horizon loop
    for k in range(N):

        if mode == "jacobi":
            for z in range(Z):
                set_XY_others(z)

        # set initial-state parameters
        for z in range(Z):
            
            if mode == "gauss-seidel":
                set_XY_others(z)
            
            opti = agents[z]["opti"]
            X = agents[z]["X"]
            U = agents[z]["U"]
            J = agents[z]["J"]
            
            xk = Xk[z].reshape(nx, 1)
            opti.set_value(agents[z]["x0"], xk)
            opti.set_initial(X, pred_X[z]) # warm start
            opti.set_initial(U, pred_U[z]) # warm start
            
            sol = opti.solve()
            X_opt = sol.value(X)
            U_opt = sol.value(U)
            
            pred_X[z] = shift_pred(X_opt)  # update shared predictions
            pred_U[z] = shift_pred(U_opt)  # update shared predictions


            uk = U_opt[:, 0].reshape((nu, 1))

            # apply first control, advance true states, shift warm starts, log
            xk_1 = xk + dt * f_np(xk, uk) #+ w[z][k, :].reshape(nx, 1)

            x_cl[z, :, k + 1] = xk_1.flatten()
            u_cl[z, :, k] = uk.flatten()
            
            Xk[z] = xk_1.flatten()
            
            J_cl[z, k] = sol.value(J)

            
    # plot
    J_cl_avg = np.mean(J_cl)
    plot(t_max, N, Z, x_cl, u_cl, x0_val, xf_val, J_cl_avg, "distributed", mode)
    
# if __name__ == "__main__":
#     # dmpc_distributed(mode="gauss-seidel")
#     dmpc_distributed(mode="jacobi")