import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import LeafSystem, BasicVector, Simulator, DynamicProgrammingOptions, FittedValueIteration



m = 1.0
I = 0.01
g = 9.81
dt = 0.05


s_max = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
steps = np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.075])

def make_axis(lo, hi, step):
    xs = np.arange(lo, hi + 1e-12, step)
    if not np.isclose(xs[np.argmin(np.abs(xs))], 0.0):
        xs = np.sort(np.append(xs, 0.0))
    return np.array(sorted(xs.tolist()))

state_axes = [make_axis(-s_max[i], s_max[i], steps[i]) for i in range(6)]


T_axis = np.linspace(-1.0,  1.0, 5)    
tau_axis = np.linspace(-0.1,  0.1, 3)
action_grid = np.array(list(itertools.product(T_axis, tau_axis)))




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
        dT, tau = (u[0]), (u[1])
        T = dT + m * g

        xdot = vx
        zdot = vz
        wdot = eps
        vxdot = -(T / m) * np.sin(w)          
        vzdot = (T / m) * np.cos(w) - g
        epsdot = tau / I

        derivatives.SetFromVector([xdot, zdot, wdot, vxdot, vzdot, epsdot])




plant = PlanarQuadrotor()
sim = Simulator(plant)


state_grid = [set(ax.tolist()) for ax in state_axes]
input_grid = [set(T_axis.tolist()), set(tau_axis.tolist())]

opts = DynamicProgrammingOptions()
opts.convergence_tol = 1e-3
opts.discount_factor = 1.0

policy_fvi, cost_to_go = FittedValueIteration(
    sim,
    lambda cont_x: (cont_x.get_continuous_state_vector().CopyToVector()  @ cont_x.get_continuous_state_vector().CopyToVector()),
    state_grid,
    input_grid,
    dt,
    opts
)


J = np.reshape(cost_to_go, [len(ax) for ax in state_axes], order="F")




def _interval_and_alpha(axis, x):
    if x <= axis[0]:
        return 0, 0, 0.0
    if x >= axis[-1]:
        j_index = len(axis) - 1
        return j_index, j_index, 0.0
    j_index = np.searchsorted(axis, x)
    i_index = j_index - 1
    i1 = j_index
    x0, x1 = axis[i_index], axis[i1]
    act = 0.0 if np.isclose(x1, x0) else (x - x0) / (x1 - x0)
    return i_index, i1, act

def V_interp(s):
    indexes = []
    alphas = []
    for d in range(6):
        i_index, i1, act = _interval_and_alpha(state_axes[d], s[d])
        indexes.append((i_index, i1))
        alphas.append(act)

    v = 0.0
    for bits in itertools.product([0, 1], repeat=6):
        w = 1.0
        corner = []
        for d, b in enumerate(bits):
            i_index, i1 = indexes[d]
            act = alphas[d]
            w *= (1 - act) if b == 0 else act
            corner.append(i_index if b == 0 else i1)
        v += w * J[tuple(corner)]
    return v




def drake_step(s, a, h):
    context = sim.get_mutable_context()
    context.SetTime(0.0)
    x = context.get_mutable_continuous_state_vector()
    x.SetFromVector(s)
    sim.get_system().get_input_port(0).FixValue(context, np.asarray(a))
    sim.Initialize()
    sim.AdvanceTo(h)
    return context.get_continuous_state_vector().CopyToVector()

def optimal_action(s):
    
    best = math.inf
    best_a = None
    for a in action_grid:
        s_next = drake_step(s, a, dt)
        curr = s @ s + V_interp(s_next)
        if curr < best:
            best = curr
            best_a = a
    return best_a

def rollout(init_state, max_steps=400, tol=1e-3, blowup=1e3):
    traj_s = [np.array(init_state, dtype=float)]
    traj_a = []
    norms = [np.linalg.norm(traj_s[0])]
    for _ in range(max_steps):
        s = traj_s[-1]
        if norms[-1] <= tol:
            break
        action_opt = optimal_action(s)
        s_next = drake_step(s, action_opt, dt)
        traj_a.append(action_opt)
        traj_s.append(s_next)
        norms.append(np.linalg.norm(s_next))
        if norms[-1] > blowup:
            break
    return np.array(traj_s), np.array(traj_a), np.array(norms)

if __name__ == "__main__":
    init_state = np.array([0.4, 0.4, 0.2, 0.05, 0.05, 0.05], dtype=float)
    S, A, norms = rollout(init_state, max_steps=500, tol=1e-3, blowup=5e2)

    print(f"Closed-loop steps: {len(S)-1}")
    print(f"Final ||s||_2: {norms[-1]:.6f}")

    
    t = np.arange(len(norms)) * dt
    plt.figure(figsize=(6, 4))
    plt.plot(t, norms, linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("||s||2")
    plt.title("Convergence to the origin under the optimal policy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/q2p3.pdf")
    plt.close()
