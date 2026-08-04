import numpy as np


def default_params():
    dt = 0.05
    steps = 100
    A = np.eye(6)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    return dt, steps, A


def default_initial_state():
    return np.array([0.0, 0.0, 0.0, 200.0, 300.0, 400.0])


def generate_trajectory(x0=None, steps=None, dt=None, A=None):
    # Fill missing parameters with defaults
    if dt is None or steps is None or A is None:
        dt_default, steps_default, A_default = default_params()
        if dt is None:
            dt = dt_default
        if steps is None:
            steps = steps_default
        if A is None:
            A = A_default

    if x0 is None:
        x0 = default_initial_state()

    trajectory = np.zeros((steps, 6))
    x_current = x0.copy()
    for k in range(steps):
        trajectory[k] = x_current
        x_current = A @ x_current

    return trajectory


def generate_target_trajectory(x0=None, steps=None, dt=None, pattern='linear', speed_scale=1.0):
    """Generate a simple target trajectory.

    Returns an array of shape (steps, 6). By default the target moves
    slowly along the Y axis with small lateral sinusoidal variation.
    """
    if dt is None or steps is None:
        dt_default, steps_default, _ = default_params()
        if dt is None:
            dt = dt_default
        if steps is None:
            steps = steps_default

    if x0 is None:
        # place target ahead of origin
        x0 = np.array([500.0, 0.0, 200.0, -20.0, 40.0, 0.0])

    traj = np.zeros((steps, 6))
    state = x0.copy()
    for k in range(steps):
        t = k * dt
        if pattern == 'linear':
            # simple constant velocity scaled by speed_scale
            traj[k] = state.copy()
            vel = state[3:] * speed_scale
            traj[k, 3:] = vel
            state[:3] = state[:3] + vel * dt
        elif pattern == 'sin':
            traj[k, :3] = state[:3] + np.array([0.0, np.sin(t) * 20.0, 0.0])
            traj[k, 3:] = state[3:] * speed_scale
        else:
            traj[k] = state

    return traj
