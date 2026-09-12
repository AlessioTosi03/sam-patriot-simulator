import numpy as np


def default_params():
    dt = 0.05
    steps = 1000
    A = np.eye(6)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    return dt, steps, A


def default_initial_state():
    return np.array([0.0, 0.0, 0.0, 300.0, 450.0, 600.0])


def generate_random_target(max_range=60000.0, min_range=20000.0, max_altitude=25000.0):
    """
    Generate a realistic Cartesian state vector [x, y, z, vx, vy, vz]
    for an endoatmospheric threat targeting the origin (0, 0, 0).
    """

    # 2. Randomize position (polar -> Cartesian)
    r_xy = np.random.uniform(min_range, max_range)
    azimuth = np.random.uniform(0, 2 * np.pi)

    x = r_xy * np.cos(azimuth)
    y = r_xy * np.sin(azimuth)

    # Ballistic missiles spawn high in the atmosphere (15 km to 25 km ceiling)
    z = np.random.uniform(15000.0, max_altitude)
    # Fast speeds (Mach 3 to Mach 5)
    speed = np.random.normal(1200.0, 200.0)
    # Steep dive angle (-45 to -75 degrees)
    gamma = np.random.uniform(-75.0, -45.0) * (np.pi / 180.0)

    # 3. Target heading (point roughly at the origin/interceptor)
    angle_to_origin = np.arctan2(-y, -x)
    heading = angle_to_origin + np.random.normal(0, 0.05)

    # 4. Calculate velocity components
    v_x = speed * np.cos(gamma) * np.cos(heading)
    v_y = speed * np.cos(gamma) * np.sin(heading)
    v_z = speed * np.sin(gamma)

    return np.array([x, y, z, v_x, v_y, v_z])


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
    object_state = x0.copy()
    for k in range(steps):
        trajectory[k] = object_state
        object_state = A @ object_state

    return trajectory


def generate_target_trajectory(x0=None, steps=None, dt=None, A=None):
    """Generate a target trajectory using the same LTI integrator as
    `generate_trajectory` so behavior and numerical integration match.

    By default the target starts high (z=2000) with an initial velocity.
    """
    # Use the same defaults/resolution as generate_trajectory
    if dt is None or steps is None or A is None:
        dt_default, steps_default, A_default = default_params()
        if dt is None:
            dt = dt_default
        if steps is None:
            steps = steps_default
        if A is None:
            A = A_default

    if x0 is None:
        x0 = generate_random_target()

    # Delegate to the same integrator implementation used for other objects.
    trajectory = generate_trajectory(x0=x0, steps=steps, dt=dt, A=A)

    # Freeze the target at the first ground-contact frame so it visibly stops at z = 0.
    impact_indices = np.where(trajectory[:, 2] <= 0.0)[0]
    if impact_indices.size > 0:
        impact_index = int(impact_indices[0])
        trajectory[impact_index:, :] = trajectory[impact_index, :]
        trajectory[impact_index, 2] = 0.0

    return trajectory


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

    if missile_x0 is None:
        # Default missile initial launch state (aiming upwards towards target)
        missile_x0 = np.array([0.0, 0.0, 0.0, 400.0, 400.0, 800.0])

    if target_x0 is None:
        target_x0 = generate_random_target(min_range=12000.0, max_range=18000.0, max_altitude=15000.0)

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

        # 1. Compute ProNav control acceleration input u(k) = a_M(k) based on LOS kinematics
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

        # Check interception condition
        if current_range <= kill_radius:
            intercepted = True
            break

        # Check target ground impact
        if target_state[2] <= 0.0:
            break

        # 2. Propagate Target State (Linear trajectory / gravity model)
        # Target experiences gravity g = [0, 0, -9.81]
        u_target = np.array([0.0, 0.0, -9.81])
        target_state = A @ target_state + B @ u_target

        # 3. Propagate Missile State using LTI update x_M(k+1) = A * x_M(k) + B * u(k)
        missile_state = A @ missile_state + B @ u_k

    return (
        np.array(missile_traj),
        np.array(target_traj),
        los_history,
        intercepted,
        min_miss_distance
    )

