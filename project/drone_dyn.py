import casadi as ca
import numpy as np

from distributed_mpc import dmpc_distributed
from decentralized_mpc import dmpc_decentralized
from centralized_mpc import dmpc_centralized

# =========================================================================
# SUPPORT FUNCTIONS
# =========================================================================

def skew_np(v):
    return np.array([[0.0,   -v[2],  v[1]],
                     [v[2],   0.0,  -v[0]],
                     [-v[1],  v[0],  0.0]])

def skew_casadi(v):
    return ca.vertcat(
        ca.hcat([0,      -v[2],   v[1]]),
        ca.hcat([v[2],    0,     -v[0]]),
        ca.hcat([-v[1],   v[0],   0])
    )

def cross_casadi(a, b):
    return ca.vertcat(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )

def eul2rotm_zyx(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])

    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])

    return Rz @ Ry @ Rx

# =========================================================================
# PRESETS
# =========================================================================

# geometry
d = 0.2        # [m] arm length

# dynamics & inertia
m = 0.5        # [kg]
Ixxb = 0.01    # [kg m^2]
Iyyb = 0.01    # [kg m^2]
Izzb = 0.05    # [kg m^2]

# initial conditions
p0 = np.array([0.5, 0.5, 1.0])        # [m]
v0 = np.array([0, 0, 0])       # [m/s]

# ZYX Euler angles [yaw, pitch, roll]
euler0 = np.array([0, 0, 0])  # [rad]

wb0 = np.array([0, 0, 0])       # [rad/s]

# simulation / MPC settings
N = 40
dt = 0.1

# number of drones
Z = 1

# safety distance for MPC
d_min = 0.2

# state and control dimensions
nx = 18       # [p(3), R(9), v(3), wb(3)]
nu = 4        # rotor thrusts F1..F4

# input bounds (thrust along +z_b, per rotor)
F_max = 20.0   # [N] choose something reasonable

# number of obstacles
no = 0
obs = np.hstack([np.random.uniform(-10, 10, (no, 2)), 10*np.ones((no, 1)), np.random.uniform(1, 5, (no, 1))])

p_target = np.array([2.0, 2.0, 2.0])
R_target = np.eye(3)
v_target = np.zeros(3)
wb_target = np.zeros(3)

init_spacing = 0.5

# cost matrices
Q = ca.DM([
    50, 50, 50,          # position
    0, 0, 0,    # rotation R
    0, 0, 0,
    0, 0, 0,
    1, 1, 1,             # linear velocity
    0.1, 0.1, 0.1        # angular velocity
])
# Q = ca.DM([
#     50, 50, 50,          # position
#     1e-3, 1e-3, 1e-3,    # rotation R
#     1e-3, 1e-3, 1e-3,    # rotation R
#     1e-3, 1e-3, 1e-3,
#     1e-3, 1e-3, 1e-3,
#     1, 1, 1,             # linear velocity
#     0.1, 0.1, 0.1        # angular velocity
# ])
Q = ca.diag(Q)
R = ca.DM(np.eye(nu))
H = 10.0 * Q


# =========================================================================
# DERIVED CONSTANTS
# =========================================================================

U_lim = [(0, F_max), (0, F_max), (0, F_max), (0, F_max)]


R0 = eul2rotm_zyx(euler0[0], euler0[1], euler0[2])

# build one-drone state
x0_single = np.hstack([p0, R0.reshape(-1), v0, wb0])

# choose a simple target: higher hover, level, zero rates
xf_single = np.hstack([p_target, R_target.reshape(-1), v_target, wb_target])

# stack initial / final for Z drones (just offsets in x)
x0_val = np.zeros((Z, nx))
xf_val = np.zeros((Z, nx))

for i in range(Z):
    offset = np.array([init_spacing*i, 0.0, 0.0])

    x0_i = x0_single.copy()
    x0_i[0:3] = p0 + offset

    xf_i = xf_single.copy()
    xf_i[0:3] = p_target + offset

    x0_val[i, :] = x0_i
    xf_val[i, :] = xf_i


# inertia matrices
Ib_np = np.diag([Ixxb, Iyyb, Izzb])
Ib_inv_np = np.linalg.inv(Ib_np)

Ib = ca.diag(ca.DM([Ixxb, Iyyb, Izzb]))
Ib_inv = ca.inv(Ib)

# gravity (world frame)
g_val = 9.81
g_vec = np.array([0.0, 0.0, -g_val])

# rotor positions in body frame
r1_np = np.array([ d, 0.0, 0.0])
r2_np = np.array([ 0.0, d, 0.0])
r3_np = np.array([-d, 0.0, 0.0])
r4_np = np.array([ 0.0,-d, 0.0])

r1 = ca.DM(r1_np)
r2 = ca.DM(r2_np)
r3 = ca.DM(r3_np)
r4 = ca.DM(r4_np)


# =========================================================================
# DYNAMICS f (CASADI)  AND f_np (NUMPY)
# =========================================================================

def f(x, u):
    """
    CasADi version MATLAB odefun, but with rotor forces as control.

    x: 18x1, ordered as [p(3); vec(R)(9); v(3); wb(3)]
    u: 4x1, scalar thrust for each rotor along body +z.
    """
    p = x[0:3]
    R = ca.reshape(x[3:12], 3, 3)
    v = x[12:15]
    wb = x[15:18]

    # rotor forces (body frame)
    F1b = ca.vertcat(0, 0, u[0])
    F2b = ca.vertcat(0, 0, u[1])
    F3b = ca.vertcat(0, 0, u[2])
    F4b = ca.vertcat(0, 0, u[3])

    # R_dot = R * skew(wb)
    R_dot = R @ skew_casadi(wb)

    # angular acceleration
    tau = (cross_casadi(r1, F1b)
         + cross_casadi(r2, F2b)
         + cross_casadi(r3, F3b)
         + cross_casadi(r4, F4b)
         - cross_casadi(wb, Ib @ wb))

    wb_dot = Ib_inv @ tau

    # total force in world frame
    Fb_total = F1b + F2b + F3b + F4b
    F_world = R @ Fb_total

    v_dot = F_world / m + ca.DM(g_vec)

    dx = ca.vertcat(
        v,                      # p_dot
        ca.reshape(R_dot, 9, 1),
        v_dot,
        wb_dot
    )

    return dx


def f_np(x, u):
    """
    Numpy version of the same dynamics, used by MPC for forward Euler.

    x: shape (18,) or (18,1)
    u: shape (4,) or (4,1)
    returns dx: shape (18,1)
    """
    x = np.array(x, dtype=float).reshape(-1)
    u = np.array(u, dtype=float).reshape(-1)

    p = x[0:3]
    R = x[3:12].reshape((3, 3))
    v = x[12:15]
    wb = x[15:18]

    F1b = np.array([0.0, 0.0, u[0]])
    F2b = np.array([0.0, 0.0, u[1]])
    F3b = np.array([0.0, 0.0, u[2]])
    F4b = np.array([0.0, 0.0, u[3]])

    R_dot = R @ skew_np(wb)

    tau = (np.cross(r1_np, F1b)
         + np.cross(r2_np, F2b)
         + np.cross(r3_np, F3b)
         + np.cross(r4_np, F4b)
         - np.cross(wb, Ib_np @ wb))

    wb_dot = Ib_inv_np @ tau

    Fb_total = F1b + F2b + F3b + F4b
    F_world = R @ Fb_total

    v_dot = F_world / m + g_vec

    dx = np.zeros(18, dtype=float)
    dx[0:3]   = v
    dx[3:12]  = R_dot.reshape(9)
    dx[12:15] = v_dot
    dx[15:18] = wb_dot

    return dx.reshape((18, 1))

# =========================================================================
# MPC CALLS
# =========================================================================

dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "gauss-seidel", "drone")
dmpc_distributed(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "jacobi", "drone")
dmpc_decentralized(Z, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "drone")
dmpc_centralized(Z, d_min, dt, N, nx, nu, U_lim, x0_val, xf_val, f, f_np, 0, obs, Q, R, H, False, "drone")
