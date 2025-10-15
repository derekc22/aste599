import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os


def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d

def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          init_qpos: np.ndarray,
          init_qvel: np.ndarray) -> None:

    mujoco.mj_resetData(m, d)
    d.qpos = init_qpos
    d.qvel = init_qvel
    mujoco.mj_forward(m, d)


def get_shifted_q(d):
    # When a system is linearized around an equilibrium point, the result is of the form:   ẋ = A(x-x̄) + B(u-ū)     (1)
    
    # However, we can define a shifted coordinate system about the equilibrium point such that x̄, ū are the new origins    
    # In this new coordinate system, the equation can instead be written as:                ẋ = Ax + Bu             (2)
    # Where, 'x' and 'u' now represent deviations from the new origins x̄ and ū, respectively - NOT the true, absolute values of 'x' and 'u'
        
    # This math works fine in our linearized model. However, if we attempt to use this model elsewhere, such as a comparison with an unlinearized model or a simulation...
    # Neither the unlinearized model nor the simulation environment will be natively aware of our shifted coordinate system
    # That is, neither will be aware that the linearized model assumes a shifted origin
    # Thus, to make the simulator consistent with the linearized model, we must explicitly shift the simulator’s state into the linearized coordinate system
        
    # Thus, instead of coding in the simulator:                                                 ẋ = Ax + Bu             (3)
    
    # We must instead implement:                                                                ẋ = A(x-x̄) + B(u-ū)     (4)
    # ...which is correctly shifted to be centered at (x̄, ū)
    # Thus, this snippet is now consistent with the linearized model where DEVIATIONS from x̄, ū enter into A and B, respectively - NOT absolute values of x, u
    
    # In practice, this means, wherever an 'x' or 'u' appear in your code implementation of the linearized model...
    # It must be replaced with an 'x-x̄' or 'u-ū', respectively
    
    # Typically, since ū = 0 at the equilibrium point (by definition), this shift will have no effect on the control logic as u-ū = u-0 = u
    # However, since x̄ is never necessarily 0, this shift will be non-trivial
    
    # Thus, we use x-π here to account for x̄ = π

    return np.array([
        d.qpos[0]-np.pi, 
        d.qvel[0], 
    ])

# Define system matrices
A = np.array([
    [0, 1],
    [1, 0]
])
B = np.array([
    [0],
    [1]
])

# Define cost matrices
Q = np.eye(2)
R = np.array([[1]])

# Define simulation time
tmax = 15


def q1e():
    
    # load model
    m, d = load_model("inverted_pendulum.xml")
    q0 = np.array([
        [np.pi+0.1],
        [0.02]
    ])
        
    viewer = mujoco.viewer.launch_passive(m, d)

    ts = int(tmax/m.opt.timestep)

    R_gains = [1, 100]
    
    for gain in R_gains:    
        
        data = np.zeros((ts, 3))
        reset(m, d, init_qpos=q0[0], init_qvel=q0[1])
        
        # Scale R
        R_scaled = gain * R
        
        # Solve for P
        P = solve_continuous_are(A, B, Q, R_scaled)
        
        # Form K_lqr
        K_lqr = np.linalg.inv(R_scaled) @ B.T @ P

        for t in range(ts):
            
            viewer.sync()
            
            # Compute control input
            q = get_shifted_q(d)
            u = -K_lqr @ q
            
            # Apply ctrl
            d.ctrl = u
            
            mujoco.mj_step(m, d)
            
            # Log data
            # Ensure that 'x' (original coordinates) is logged, not 'x-x̄' (shifted coordinates) by adding x̄ back to the shifted coordinates
            data[t] = np.concatenate([q + [np.pi, 0], u])
            
        # Plot
        t_data = np.arange(0, tmax, m.opt.timestep)

        plt.subplot(3, 1, 1)
        plt.plot(t_data, data[:, 0], label=f"gain={gain}")
        plt.xlabel("t [s]")
        plt.ylabel("θ [rad]")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t_data, data[:, 1], label=f"gain={gain}")
        plt.xlabel("t [s]")
        plt.ylabel("θ_dot [rad/s]")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t_data, data[:, 2], label=f"gain={gain}")
        plt.xlabel("t [s]")
        plt.ylabel("u [Nm]")
        plt.legend()
        plt.grid(True)

    plt.suptitle("LQR Control with varying R gain")
    plt.tight_layout()
    plt.savefig("plots/q1e.pdf")
    plt.close()

    viewer.close()


def q1g():
    
    # load model
    m, d = load_model("inverted_pendulum.xml")
    q0s = np.array([
        [[np.pi+0.1],
         [0.02]],
         
       [[np.pi+1],
        [0.1]]
    ])
        
    viewer = mujoco.viewer.launch_passive(m, d)

    ts = int(tmax/m.opt.timestep)
    
    ic_strings = [ ["π+0.1", "0.02"], ["π+1", "0.1"]]
    
    for i, q0 in enumerate(q0s):    
        
        data = np.zeros((ts, 3))
        reset(m, d, init_qpos=q0[0], init_qvel=q0[1])
        
        # Solve for P
        P = solve_continuous_are(A, B, Q, R)
        
        # Form K_lqr
        K_lqr = np.linalg.inv(R) @ B.T @ P

        for t in range(ts):
            
            viewer.sync()
            
            # Compute control input
            q = get_shifted_q(d)
            u = -K_lqr @ q
            
            # Apply ctrl
            d.ctrl = u
            
            mujoco.mj_step(m, d)
            
            # Log data
            # Ensure that 'x' (original coordinates) is logged, not 'x-x̄' (shifted coordinates) by adding x̄ back to the shifted coordinates
            data[t] = np.concatenate([q + [np.pi, 0], u])
            
        # Plot
        t_data = np.arange(0, tmax, m.opt.timestep)

        plt.subplot(3, 1, 1)
        plt.plot(t_data, data[:, 0], label=f"θ₀={ic_strings[i][0]}, θdot₀={ic_strings[i][1]}")
        plt.xlabel("t [s]")
        plt.ylabel("θ [rad]")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t_data, data[:, 1], label=f"θ₀={ic_strings[i][0]}, θdot₀={ic_strings[i][1]}")
        plt.xlabel("t [s]")
        plt.ylabel("θ_dot [rad/s]")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t_data, data[:, 2], label=f"θ₀={ic_strings[i][0]}, θdot₀={ic_strings[i][1]}")
        plt.xlabel("t [s]")
        plt.ylabel("u [Nm]")
        plt.legend()
        plt.grid(True)

    plt.suptitle("LQR Control with varying initial condition")
    plt.tight_layout()
    plt.savefig("plots/q1g.pdf")
    plt.close()
    
    viewer.close()



if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    q1e()
    time.sleep(1)
    q1g()