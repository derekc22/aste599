# Requires: pydrake, numpy, matplotlib
import numpy as np
from pydrake.all import (
    LeafSystem, BasicVector, DiagramBuilder, Simulator,
    DynamicProgrammingOptions, FittedValueIteration
)
import matplotlib.pyplot as plt

# Constants
m = 1.0
I = 0.01
g = 9.81
dt = 0.05

# Quadrotor LeafSystem with 6 states, 2 inputs
class PlanarQuadrotor(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareContinuousState(6)
        self.DeclareVectorInputPort("u", BasicVector(2))
        self.DeclareVectorOutputPort("y", BasicVector(6), self._do_output)

    def _do_output(self, context, y):
        y.SetFromVector(context.get_continuous_state_vector().CopyToVector())

    def DoCalcTimeDerivatives(self, context, derivatives):
        x, z, w, vx, vz, eps = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port(0).Eval(context)
        T, tau = u[0], u[1]

        xdot = vx
        zdot = vz
        wdot = eps
        vxdot = (T/m)*np.sin(w)
        vzdot = (T/m)*np.cos(w) - g
        epsdot = tau/I

        derivatives.SetFromVector([xdot, zdot, wdot, vxdot, vzdot, epsdot])

# Drake cost function: returns instantaneous running cost l(x,u)
def running_cost(context):
    s = context.get_continuous_state_vector().CopyToVector()
    return float(s @ s)  # Q = I

# Build simulator on the system (required by FVI)
plant = PlanarQuadrotor()
sim = Simulator(plant)

# State grids (each as a Python set of floats)
def make_grid(lo, hi, step):
    # Ensure origin is included and endpoints align
    grid = np.arange(lo, hi + 1e-12, step)
    if not np.isclose(grid[np.argmin(np.abs(grid))], 0.0):
        grid = np.sort(np.append(grid, 0.0))
    return set(grid.tolist())

s_max = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
steps = np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.075])

state_grid = [
    make_grid(-s_max[i], s_max[i], steps[i]) for i in range(6)
]

# Action grids
T_grid = set(np.linspace(-1.0, 1.0, 5))      # step 0.5
tau_grid = set(np.linspace(-0.1, 0.1, 3))    # step 0.1
input_grid = [T_grid, tau_grid]

# DP options
opts = DynamicProgrammingOptions()
opts.convergence_tol = 1e-3
opts.discount_factor = 0.995  # contraction with quadratic running cost
# Optional: monitor iterations by setting a visualization callback, or store histories.

# Run value iteration
policy, cost_to_go = FittedValueIteration(
    sim, running_cost, state_grid, input_grid, dt, opts
)

# Helper: reshape J onto an ndarray with axes matching the grids
# Note: cost_to_go is returned in column-major order over the mesh.
grid_sizes = [len(g) for g in state_grid]
J = np.reshape(cost_to_go, grid_sizes, order="F")

# Convergence verification: the library stops when sup-norm change < tol.
# If you recorded callbacks with intermediate J, you could plot ||J^{k+1}-J^k||.
print("Value iteration complete. J shape:", J.shape)

# Visualization: average value over nuisance dimensions to get 2D views
axes = [0, 1]  # for example, x vs z
all_axes = list(range(6))
reduce_axes = tuple(a for a in all_axes if a not in axes)
J_xz = np.mean(J, axis=reduce_axes)

# Build sorted arrays for plotting
def sorted_array(s): 
    arr = np.array(sorted(list(s)))
    return arr

x_axis = sorted_array(state_grid[0])
z_axis = sorted_array(state_grid[1])

plt.figure(figsize=(6,5))
plt.title("Average value over other states, V(x,z)")
plt.xlabel("x")
plt.ylabel("z")
plt.imshow(
    np.flipud(J_xz), 
    extent=(x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]),
    aspect="auto"
)
plt.colorbar(label="V")
plt.savefig("q2p2.pdf")
plt.show()
