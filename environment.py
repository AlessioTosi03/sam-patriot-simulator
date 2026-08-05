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
