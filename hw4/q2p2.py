import numpy as np
from pydrake.all import LeafSystem, BasicVector, DiagramBuilder, Simulator, DynamicProgrammingOptions, FittedValueIteration
import matplotlib.pyplot as plt
import os


m = 1.0
I = 0.01
g = 9.81
dt = 0.05


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
        
        hover_thrust = m * g
        T = u[0] + hover_thrust  
        tau = u[1]

        xdot = vx
        zdot = vz
        wdot = eps
        vxdot = -(T/m)*np.sin(w)
        vzdot = (T/m)*np.cos(w) - g
        epsdot = tau/I

        derivatives.SetFromVector([xdot, zdot, wdot, vxdot, vzdot, epsdot])


def running_cost(context):
    s = context.get_continuous_state_vector().CopyToVector()
    return s @ s


plant = PlanarQuadrotor()
sim = Simulator(plant)


def make_grid(lo, hi, step):
    
    grid = np.arange(lo, hi + 1e-12, step)
    if not np.isclose(grid[np.argmin(np.abs(grid))], 0.0):
        grid = np.sort(np.append(grid, 0.0))
    return set(grid.tolist())

s_max = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
steps = np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.075])

state_grid = [make_grid(-s_max[i], s_max[i], steps[i]) for i in range(6)]


T_grid = set(np.linspace(-1.0, 1.0, 5))      
tau_grid = set(np.linspace(-0.1, 0.1, 3))    
input_grid = [T_grid, tau_grid]


opts = DynamicProgrammingOptions()
opts.convergence_tol = 1e-3
opts.discount_factor = 1 



policy, cost_to_go = FittedValueIteration(sim, running_cost, state_grid, input_grid, dt, opts)



grid_sizes = [len(g) for g in state_grid]
J = np.reshape(cost_to_go, grid_sizes, order="F")



print("Value iteration complete. J shape:", J.shape)


axes = [0, 1]  
all_axes = list(range(6))
reduce_axes = tuple(a for a in all_axes if a not in axes)
J_xz = np.mean(J, axis=reduce_axes)


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
os.makedirs("hw4/plots", exist_ok=True)
plt.savefig("hw4/plots/q2p2.pdf")

plt.close()
