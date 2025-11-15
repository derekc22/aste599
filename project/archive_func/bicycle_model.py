import casadi as ca    
import numpy as np
from distributed_mpc import dmpc_distributed
from decentralized_mpc import dmpc_decentralized
from centralized_mpc import dmpc_centralized


# bicycle model

# number of agents
Z = 3

# hard separation distance
d_min = 0.01

# discretization
dt = 0.5
N = 40

# define state, control dim
nx = 4
nu = 2

# control constraints [delta, a]
U_lim = [0.7, 0.2]

x0_val = np.hstack([np.random.randint(-10, 10, (Z, 2)), np.zeros((Z, nx-2))])
xf_val = np.hstack([np.random.randint(10, 11, (Z, 2)), np.zeros((Z, nx-2))])

# kinematics, forward Euler integration in constraints
def f(x, u):
    L = 1
    theta, v = x[2], x[3]
    delta, a = u[0], u[1]
    dx = v * ca.cos(theta)
    dy = v * ca.sin(theta)
    dtheta = (v / L) * delta
    dv = a
    return ca.vcat([dx, dy, dtheta, dv])

# non-CasADi dynamics for closed-loop propagation
def f_np(x, u):
    L = 1
    theta, v = x[2], x[3]
    delta, a = u[0], u[1]
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = (v / L) * delta
    dv = a
    return np.array([dx, dy, dtheta, dv], dtype=float)


dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, "gauss-seidel")
dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, "jacobi")
dmpc_decentralized(Z, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0)
dmpc_centralized(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0)



