import numpy as np
from scipy.linalg import solve_continuous_are


def q1d():
    
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

    # Solve for P
    P = solve_continuous_are(A, B, Q, R)
    
    # Form K_lqr
    K_lqr = np.linalg.inv(R) @ B.T @ P

    print(K_lqr)


if __name__ == "__main__":
    q1d()