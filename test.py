import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import os

def main():


    # number of agents
    Z = 3
    
    # hard separation distance
    d_min = 0.01

    # geometric
    L = 1

    # discretize
    dt = 0.5
    N = 40
    t_max = N * dt

    # define state, control dim
    nx = 4
    nu = 2

    # control constraints [delta, a]
    U_lim = [0.7, 0.2]

    # initial and goal states, shape (Z, nx)
    x0_val = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
    ], dtype=float)
    xf_val = np.array([
        [9.1, 5, 0, 0],
        [9, 5, 0, 0],
        [8.9, 5, 0, 0],
    ], dtype=float)

    # cost matrices
    Q = ca.DM(np.eye(nx))
    R = ca.DM(np.eye(nu))
    H = ca.DM(np.eye(nx))

    # kinematics, forward Euler integration in constraints
    def f(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return ca.vcat([dx, dy, dtheta, dv])

    # non-CasADi dynamics for closed-loop propagation
    def f_np(x, u):
        theta, v = x[2], x[3]
        delta, a = u[0], u[1]
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = (v / L) * delta
        dv = a
        return np.array([dx, dy, dtheta, dv], dtype=float)

    # pre-sample disturbances for each agent, shape list[Z] of (N, nx)
    # w = [np.random.multivariate_normal(np.zeros(nx), np.diag([0.001] * nx), N) for _ in range(Z)]
    w = [np.random.multivariate_normal(np.zeros(nx), np.diag([0.00] * nx), N) for _ in range(Z)]

    # build a local OCP for one agent, with other agents' XY as parameters
    def build_agent_opti():
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        X0 = opti.parameter(nx, 1)
        Xf = opti.parameter(nx, 1)
        # other agents' predicted positions over horizon
        XY_others = [opti.parameter(2, N + 1) for _ in range(Z - 1)]

        J = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            J += ca.mtimes([(xk - Xf).T, Q, (xk - Xf)]) + ca.mtimes([uk.T, R, uk])
            x_next = xk + dt * f(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)
            # collision avoidance with others
            for XYj in XY_others:
                opti.subject_to(ca.sumsqr(X[0:2, k] - XYj[:, k]) >= d_min ** 2)

        # terminal cost
        xN = X[:, N]
        J += ca.mtimes([(xN - Xf).T, H, (xN - Xf)])

        # input bounds and initial condition
        opti.subject_to(X[:, 0] == X0)
        opti.subject_to(opti.bounded(-U_lim[0], U[0, :], U_lim[0]))
        opti.subject_to(opti.bounded(-U_lim[1], U[1, :], U_lim[1]))

        # crater avoidance, center (2,2), radius 1
        opti.subject_to((X[0, :] - 2) ** 2 + (X[1, :] - 2) ** 2 >= 1.0)

        opti.minimize(J)
        opti.solver("ipopt")
        return {"opti": opti, "X": X, "U": U, "X0": X0, "Xf": Xf, "XY_others": XY_others}

    # build agents
    agents = [build_agent_opti() for _ in range(Z)]
    for z in range(Z):
        agents[z]["opti"].set_value(agents[z]["Xf"], ca.DM(xf_val[z, :]).reshape((nx, 1)))

    # straight-line XY warm starts, zero theta, zero v
    pred_X = []
    for z in range(Z):
        x0z = ca.DM(x0_val[z, :]).reshape((nx, 1))
        xfz = ca.DM(xf_val[z, :]).reshape((nx, 1))
        XY = ca.hcat([x0z[0:2] + (k / float(N)) * (xfz[0:2] - x0z[0:2]) for k in range(N + 1)])
        theta = ca.DM.zeros(1, N + 1)
        v = ca.DM.zeros(1, N + 1)
        pred_X.append(ca.vertcat(XY, theta, v))

    # helpers
    def set_XY_others(z):
        idx = 0
        for j in range(Z):
            if j == z:
                continue
            agents[z]["opti"].set_value(agents[z]["XY_others"][idx], pred_X[j][0:2, :])
            idx += 1

    def shift_prediction(Xdm):
        return ca.hcat([Xdm[:, 1:], Xdm[:, -1:]])

    # logs for plotting
    x_cl = np.zeros((Z, N + 1, nx), dtype=float)
    u_cl = np.zeros((Z, N, nu), dtype=float)
    for z in range(Z):
        x_cl[z, 0, :] = x0_val[z, :]

    # true states as column vectors
    x_true = [x0_val[z, :].reshape(nx, 1).astype(float) for z in range(Z)]

    # receding-horizon loop
    for k in range(N):
        # set initial-state parameters
        for z in range(Z):
            agents[z]["opti"].set_value(agents[z]["X0"], ca.DM(x_true[z]).reshape((nx, 1)))

        # push current predicted XY for coupling
        for z in range(Z):
            set_XY_others(z)

        # sequential solves, fixed order z = 0..Z-1
        X_solutions = [None] * Z
        U_solutions = [None] * Z
        for z in range(Z):
            opti = agents[z]["opti"]
            X = agents[z]["X"]
            U = agents[z]["U"]
            opti.set_initial(X, pred_X[z])
            opti.set_initial(U, ca.DM.zeros(nu, N))
            sol = opti.solve()
            X_solutions[z] = sol.value(X)
            U_solutions[z] = sol.value(U)
            pred_X[z] = ca.DM(X_solutions[z])  # update shared predictions

        # apply first control, advance true states, shift warm starts, log
        for z in range(Z):
            u0 = U_solutions[z][:, 0]
            x0 = x_true[z].flatten()
            x1 = x0 + dt * f_np(x0, u0) + w[z][k, :]
            x_true[z] = x1.reshape(nx, 1)
            x_cl[z, k + 1, :] = x1
            u_cl[z, k, :] = u0
            X_shifted = shift_prediction(pred_X[z])
            X_shifted[:, 0] = ca.DM(x_true[z])
            pred_X[z] = X_shifted

    # plot
    os.makedirs("plots", exist_ok=True)
    t = np.linspace(0, t_max, N + 1)

    plt.figure(figsize=(9, 11))
    plt.subplot(5, 1, 1)
    for z in range(Z):
        plt.plot(x_cl[z, :, 0], x_cl[z, :, 1], label=f'agent {z} traj')
        plt.plot(x0_val[z, 0], x0_val[z, 1], 'go')
        plt.plot(xf_val[z, 0], xf_val[z, 1], 'ro')
    theta = np.linspace(0, 2 * np.pi, 100)
    crater_x = 2 + 1.0 * np.cos(theta)
    crater_y = 2 + 1.0 * np.sin(theta)
    plt.plot(crater_x, crater_y, 'k--', label='crater')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()

    plt.subplot(5, 1, 2)
    for z in range(Z):
        plt.plot(t, x_cl[z, :, 2])
    plt.xlabel('t [s]')
    plt.ylabel('θ [rad]')
    plt.grid()

    plt.subplot(5, 1, 3)
    for z in range(Z):
        plt.plot(t, x_cl[z, :, 3])
    plt.ylabel('v [ms⁻¹]')
    plt.xlabel('t [s]')
    plt.grid()

    plt.subplot(5, 1, 4)
    for z in range(Z):
        plt.step(t[:-1], u_cl[z, :, 0], where='post')
    plt.ylabel('δ [rad]')
    plt.xlabel('t [s]')
    plt.grid()

    plt.subplot(5, 1, 5)
    for z in range(Z):
        plt.step(t[:-1], u_cl[z, :, 1], where='post')
    plt.ylabel('a [ms⁻²]')
    plt.xlabel('t [s]')
    plt.grid()

    plt.suptitle('distributed mpc')
    plt.tight_layout()
    plt.savefig("project/plots/distributed_mpc.pdf")
    plt.close()


if __name__ == "__main__":
    main()