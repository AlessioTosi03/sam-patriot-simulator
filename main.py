from environment import generate_trajectory, default_initial_state, default_params, generate_target_trajectory
from visualization import animate_trajectory


def run():
    dt, steps, _ = default_params()
    x0 = default_initial_state()
    trajectory = generate_trajectory(x0=x0, steps=steps)
    target_traj = generate_target_trajectory(steps=steps)
    # interval in milliseconds
    interval_ms = int(dt * 1000)
    animate_trajectory(trajectory, target_trajectory=target_traj, interval=interval_ms)


if __name__ == '__main__':
    run()
