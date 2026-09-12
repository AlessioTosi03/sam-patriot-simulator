import numpy as np
from environment import simulate_interception, default_params, generate_random_target
from visualization import animate_trajectory


def run():
    dt = 0.05
    steps = 1000

    # Threat initial state (incoming ballistic target within 12-18 km engagement zone)
    target_x0 = generate_random_target(min_range=12000.0, max_range=18000.0, max_altitude=15000.0)

    # Patriot Interceptor initial launch state
    missile_x0 = np.array([0.0, 0.0, 0.0, 400.0, 400.0, 800.0])

    print("=" * 70)
    print("PATRIOT SAM SIMULATOR - LOS & PRONAV GUIDANCE SIMULATION")
    print("=" * 70)
    print(f"Target Initial State:  Pos [{target_x0[0]:.1f}, {target_x0[1]:.1f}, {target_x0[2]:.1f}] m")
    print(f"                       Vel [{target_x0[3]:.1f}, {target_x0[4]:.1f}, {target_x0[5]:.1f}] m/s")
    print(f"Missile Initial State: Pos [{missile_x0[0]:.1f}, {missile_x0[1]:.1f}, {missile_x0[2]:.1f}] m")
    print(f"                       Vel [{missile_x0[3]:.1f}, {missile_x0[4]:.1f}, {missile_x0[5]:.1f}] m/s")
    print("-" * 70)

    missile_traj, target_traj, los_hist, intercepted, min_miss = simulate_interception(
        missile_x0=missile_x0,
        target_x0=target_x0,
        steps=steps,
        dt=dt,
        N=4.0,
        kill_radius=15.0
    )

    k_final = len(missile_traj)
    time_final = k_final * dt
    initial_los = los_hist[0]
    final_los = los_hist[-1]

    print(f"Simulation completed in {k_final} steps ({time_final:.2f} s)")
    print(f"Initial Range (R_0):          {initial_los.range:.2f} m")
    print(f"Initial Closing Velocity:      {initial_los.v_closing:.2f} m/s")
    print(f"Final Range (Miss Distance):  {final_los.range:.2f} m")
    print(f"Minimum Miss Distance:        {min_miss:.2f} m")
    print(f"Target Intercepted:           {'YES' if intercepted else 'NO'}")
    print("=" * 70)

    interval_ms = int(dt * 1000)
    animate_trajectory(
        missile_traj,
        target_trajectory=target_traj,
        interval=interval_ms,
        title=f"Patriot SAM: ProNav Interception (Miss Distance: {min_miss:.1f} m)"
    )


if __name__ == '__main__':
    run()

