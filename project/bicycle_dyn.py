import casadi as ca    
import numpy as np
from distributed_mpc import dmpc_distributed
from decentralized_mpc import dmpc_decentralized
from centralized_mpc import dmpc_centralized


# bicycle model

# number of agents
Z = 5

# hard separation distance
d_min = 0.01

# discretization
dt = 0.5
N = 40

# define state, control dim
nx = 5
nu = 2

# control constraints [delta, a]
U_lim = [(-0.7, 0.7), (-0.2, 0.2)]

x0_val = np.hstack([np.random.uniform(-10, 10, (Z, 2)), np.zeros((Z, nx-2))])
xf_val = np.hstack([np.random.uniform(10, 11, (Z, 2)), np.zeros((Z, nx-2))])

# number of obstacles
no = 8
obs = np.hstack([np.random.uniform(-10, 10, (no, 2)), 2*np.ones((no, 1)), np.random.uniform(1, 5, (no, 1))])

# kinematics, forward Euler integration in constraints
def f(x, u):
    L = 1
    theta, v = x[3], x[4]
    delta, a = u[0], u[1]
    dx = v * ca.cos(theta)
    dy = v * ca.sin(theta)
    dz = [0]
    dtheta = (v / L) * delta
    dv = a
    return ca.vcat([dx, dy, dz, dtheta, dv])

# non-CasADi dynamics for closed-loop propagation
def f_np(x, u):
    L = 1
    theta, v = x[3], x[4]
    delta, a = u[0], u[1]
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dz = [0]
    dtheta = (v / L) * delta
    dv = a
    return np.array([dx, dy, dz, dtheta, dv], dtype=float)

Q = ca.DM(np.eye(nx))
R = ca.DM(np.eye(nu))
H = ca.DM(np.eye(nx))

dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "bicycle")
# dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, True, "jacobi", "bicycle")
# dmpc_decentralized(Z, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, True, "bicycle")
# dmpc_centralized(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, True, "bicycle")



