import numpy as np


def generate_random_target(max_range=60000.0, min_range=20000.0, max_altitude=25000.0):
    """
    Generate a realistic Cartesian state vector [x, y, z, vx, vy, vz]
    for an endoatmospheric threat targeting the origin (0, 0, 0).
    """

    # Position and velocity in spherical/polar coordinates
    r_xy = np.random.uniform(min_range, max_range)
    azimuth = np.random.uniform(0, 2 * np.pi)
    x = r_xy * np.cos(azimuth)
    y = r_xy * np.sin(azimuth)
    z = np.random.uniform(15000.0, max_altitude)

    # Ballistic trajectory pointing towards origin
    speed = np.random.normal(1200.0, 200.0)
    gamma = np.random.uniform(-75.0, -45.0) * (np.pi / 180.0)
    angle_to_origin = np.arctan2(-y, -x)
    heading = angle_to_origin + np.random.normal(0, 0.05)

    v_x = speed * np.cos(gamma) * np.cos(heading)
    v_y = speed * np.cos(gamma) * np.sin(heading)
    v_z = speed * np.sin(gamma)

    return np.array([x, y, z, v_x, v_y, v_z])


def get_lti_matrices(dt: float):
    """
    Returns discrete-time LTI state transition matrix A (6x6)
    and control input matrix B (6x3) for 3D state vector [x, y, z, vx, vy, vz]^T
    and acceleration control input u = [ax, ay, az]^T:
    
        x(k+1) = A * x(k) + B * u(k)
    """
    A = np.eye(6)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt

    B = np.zeros((6, 3))
    B[0, 0] = 0.5 * (dt ** 2)
    B[1, 1] = 0.5 * (dt ** 2)
    B[2, 2] = 0.5 * (dt ** 2)
    B[3, 0] = dt
    B[4, 1] = dt
    B[5, 2] = dt

    return A, B


def simulate_interception(
    missile_x0=None,
    target_x0=None,
    steps=1000,
    dt=0.05,
    N=3.5,
    max_accel=300.0,
    kill_radius=15.0
):
    """
    Simulates closed-loop interception between Patriot missile and target
    using discrete-time LTI state-space dynamics and ProNav guidance.
    
    Returns:
        missile_traj: np.ndarray of shape (k_final, 6)
        target_traj:  np.ndarray of shape (k_final, 6)
        los_history:  list of LOSState for each step
        intercepted:  bool (True if miss distance <= kill_radius)
        min_miss_distance: float (minimum distance reached in meters)
    """
    from guidance import compute_pronav_acceleration

    if target_x0 is None:
        target_x0 = generate_random_target(min_range=12000.0, max_range=18000.0, max_altitude=15000.0)

    if missile_x0 is None:
        # Orient launcher towards detected target azimuth
        r_xy = float(np.hypot(target_x0[0], target_x0[1]))
        dir_x = target_x0[0] / r_xy if r_xy > 1e-3 else 0.707
        dir_y = target_x0[1] / r_xy if r_xy > 1e-3 else 0.707
        missile_x0 = np.array([0.0, 0.0, 0.0, 500.0 * dir_x, 500.0 * dir_y, 700.0])

    A, B = get_lti_matrices(dt)

    missile_state = missile_x0.copy()
    target_state = target_x0.copy()

    missile_traj = []
    target_traj = []
    los_history = []

    min_miss_distance = float('inf')
    intercepted = False

    for k in range(steps):
        missile_traj.append(missile_state.copy())
        target_traj.append(target_state.copy())

        # Guidance command via ProNav
        u_k, los_k = compute_pronav_acceleration(
            target_state=target_state,
            missile_state=missile_state,
            N=N,
            max_accel=max_accel,
            include_gravity=True
        )
        los_history.append(los_k)

        current_range = los_k.range
        if current_range < min_miss_distance:
            min_miss_distance = current_range

        if current_range <= kill_radius:
            intercepted = True
            break

        if target_state[2] <= 0.0:
            break

        # State propagation under LTI discrete dynamics
        u_target = np.array([0.0, 0.0, -9.81])
        target_state = A @ target_state + B @ u_target
        missile_state = A @ missile_state + B @ u_k

    return (
        np.array(missile_traj),
        np.array(target_traj),
        los_history,
        intercepted,
        min_miss_distance
    )

