# def main():

#     import numpy as np
#     import casadi as ca
#     import matplotlib.pyplot as plt
#     import os

#     # number of agents
#     Z = 2

#     # geometry
#     L = 1

#     # discretization
#     dt = 0.5
#     N = 40
#     t_max = N * dt

#     # dimensions
#     nx = 4
#     nu = 2

#     # input limits [delta, a]
#     U_lim = [0.7, 0.2]

#     # initial and goal states, shape (Z, nx)
#     x0_val = np.array([
#         [0, 0, 0, 0],
#         [5, 0, 0, 0]
#     ], dtype=float)
#     xf_val = np.array([
#         [5, 3, 0, 0],
#         [10, 6, 0, 0]
#     ], dtype=float)

#     # cost matrices
#     Q = ca.DM(np.eye(nx))
#     R = ca.DM(np.eye(nu))
#     H = ca.DM(np.eye(nx))

#     # kinematics, forward Euler inside constraints
#     def f(x, u):
#         theta, v = x[2], x[3]
#         delta, a = u[0], u[1]
#         dx = v * ca.cos(theta)
#         dy = v * ca.sin(theta)
#         dtheta = (v / L) * delta
#         dv = a
#         return ca.vcat([dx, dy, dtheta, dv])

#     # numpy dynamics for propagation
#     def f_np(x, u):
#         theta, v = x[2], x[3]
#         delta, a = u[0], u[1]
#         dx = v * np.cos(theta)
#         dy = v * np.sin(theta)
#         dtheta = (v / L) * delta
#         dv = a
#         return np.array([dx, dy, dtheta, dv], dtype=float)

#     # disturbances, per agent
#     w = [np.random.multivariate_normal(np.zeros(nx), np.diag([0.000] * nx), N) for _ in range(Z)]

#     # hard separation distance
#     d_min = 0.01

#     # centralized Opti
#     opti = ca.Opti()

#     # decision variables per agent
#     X = [opti.variable(nx, N + 1) for _ in range(Z)]
#     U = [opti.variable(nu, N) for _ in range(Z)]

#     # parameters per agent
#     X0 = [opti.parameter(nx, 1) for _ in range(Z)]
#     XF = [opti.parameter(nx, 1) for _ in range(Z)]
#     for z in range(Z):
#         opti.set_value(XF[z], ca.DM(xf_val[z, :]).reshape((nx, 1)))

#     # objective and dynamics
#     J = 0
#     for z in range(Z):
#         for k in range(N):
#             xk = X[z][:, k]
#             uk = U[z][:, k]
#             J += ca.mtimes([(xk - XF[z]).T, Q, (xk - XF[z])]) + ca.mtimes([uk.T, R, uk])
#             x_next = xk + dt * f(xk, uk)
#             opti.subject_to(X[z][:, k + 1] == x_next)
#         xN = X[z][:, N]
#         J += ca.mtimes([(xN - XF[z]).T, H, (xN - XF[z])])

#     # constraints
#     for z in range(Z):
#         # initial condition
#         opti.subject_to(X[z][:, 0] == X0[z])
#         # input bounds
#         opti.subject_to(opti.bounded(-U_lim[0], U[z][0, :], U_lim[0]))
#         opti.subject_to(opti.bounded(-U_lim[1], U[z][1, :], U_lim[1]))
#         # crater avoidance, center (2,2), radius 1
#         opti.subject_to((X[z][0, :] - 2) ** 2 + (X[z][1, :] - 2) ** 2 >= 1.0)

#     # pairwise collision avoidance for all k
#     for k in range(N + 1):
#         for i in range(Z):
#             for j in range(i + 1, Z):
#                 opti.subject_to(ca.sumsqr(X[i][0:2, k] - X[j][0:2, k]) >= d_min ** 2)

#     # objective and solver
#     opti.minimize(J)
#     opti.solver("ipopt")

#     # solve function, sets all X0 parameters, returns all U solutions
#     def solve(x0_mat):
#         for z in range(Z):
#             opti.set_value(X0[z], ca.DM(x0_mat[z, :]).reshape((nx, 1)))
#         sol = opti.solve()
#         U_solutions = [sol.value(U[z]) for z in range(Z)]
#         return U_solutions

#     # closed-loop storage
#     x_cl = np.zeros((Z, N + 1, nx), dtype=float)
#     u_cl = np.zeros((Z, N, nu), dtype=float)
#     for z in range(Z):
#         x_cl[z, 0, :] = x0_val[z, :]

#     # true states
#     xk = [x0_val[z, :].reshape(nx,).astype(float) for z in range(Z)]

#     # receding-horizon loop
#     for k in range(N):
#         U_sol_list = solve(np.vstack(xk))
#         # apply first input, propagate, log
#         for z in range(Z):
#             uk = U_sol_list[z][:, 0]
#             xk1 = xk[z] + dt * f_np(xk[z], uk) + w[z][k, :]
#             x_cl[z, k + 1, :] = xk1
#             u_cl[z, k, :] = uk
#             xk[z] = xk1

#     # plot
#     os.makedirs("plots", exist_ok=True)
#     t = np.linspace(0, t_max, N + 1)

#     plt.figure(figsize=(9, 11))
#     plt.subplot(5, 1, 1)
#     for z in range(Z):
#         plt.plot(x_cl[z, :, 0], x_cl[z, :, 1], label=f'agent {z} traj')
#         plt.plot(x0_val[z, 0], x0_val[z, 1], 'go')
#         plt.plot(xf_val[z, 0], xf_val[z, 1], 'ro')
#     theta = np.linspace(0, 2 * np.pi, 100)
#     crater_x = 2 + 1.0 * np.cos(theta)
#     crater_y = 2 + 1.0 * np.sin(theta)
#     plt.plot(crater_x, crater_y, 'k--', label='crater')
#     plt.xlabel('x [m]')
#     plt.ylabel('y [m]')
#     plt.grid()
#     plt.legend()

#     plt.subplot(5, 1, 2)
#     for z in range(Z):
#         plt.plot(t, x_cl[z, :, 2])
#     plt.xlabel('t [s]')
#     plt.ylabel('θ [rad]')
#     plt.grid()

#     plt.subplot(5, 1, 3)
#     for z in range(Z):
#         plt.plot(t, x_cl[z, :, 3])
#     plt.ylabel('v [ms⁻¹]')
#     plt.xlabel('t [s]')
#     plt.grid()

#     plt.subplot(5, 1, 4)
#     for z in range(Z):
#         plt.step(t[:-1], u_cl[z, :, 0], where='post')
#     plt.ylabel('δ [rad]')
#     plt.xlabel('t [s]')
#     plt.grid()

#     plt.subplot(5, 1, 5)
#     for z in range(Z):
#         plt.step(t[:-1], u_cl[z, :, 1], where='post')
#     plt.ylabel('a [ms⁻²]')
#     plt.xlabel('t [s]')
#     plt.grid()

#     plt.suptitle('Centralized multiagent MPC, hard constraints')
#     plt.tight_layout()
#     plt.savefig("project/plots/distributed.pdf")
#     plt.close()


# if __name__ == "__main__":
#     main()