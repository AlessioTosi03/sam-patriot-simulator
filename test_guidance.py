import unittest
import numpy as np
from guidance import compute_los, compute_pronav_acceleration


class TestGuidance(unittest.TestCase):

    def test_los_computation_basic(self):
        """Verifies collinear closing kinematics (range, unit vector, closing speed, zero LOS rate)."""
        target_state = np.array([1000.0, 0.0, 0.0, -200.0, 0.0, 0.0])
        missile_state = np.array([0.0, 0.0, 0.0, 500.0, 0.0, 0.0])

        los = compute_los(target_state, missile_state)

        self.assertAlmostEqual(los.range, 1000.0)
        np.testing.assert_allclose(los.u_r, np.array([1.0, 0.0, 0.0]), atol=1e-5)
        np.testing.assert_allclose(los.v_rel, np.array([-700.0, 0.0, 0.0]), atol=1e-5)
        self.assertAlmostEqual(los.v_closing, 700.0)
        np.testing.assert_allclose(los.omega_los, np.zeros(3), atol=1e-5)

    def test_los_angular_rate(self):
        """Verifies non-zero cross-line angular rate calculation."""
        target_state = np.array([1000.0, 0.0, 0.0, 0.0, 200.0, 0.0])
        missile_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        los = compute_los(target_state, missile_state)

        expected_omega = np.array([0.0, 0.0, 0.2])
        np.testing.assert_allclose(los.omega_los, expected_omega, atol=1e-5)

    def test_pronav_acceleration_command(self):
        """Verifies ProNav acceleration vector is non-zero and perpendicular to LOS."""
        target_state = np.array([10000.0, 1000.0, 0.0, -300.0, 0.0, 0.0])
        missile_state = np.array([0.0, 0.0, 0.0, 1000.0, 0.0, 0.0])

        accel_cmd, los = compute_pronav_acceleration(
            target_state, missile_state, N=3.0, include_gravity=False
        )

        self.assertAlmostEqual(float(np.dot(accel_cmd, los.u_r)), 0.0, places=3)
        self.assertGreater(float(np.linalg.norm(accel_cmd)), 0.0)

    def test_interception_simulation(self):
        """Verifies 3D closed-loop interception achieves miss distance within kill radius."""
        from environment import simulate_interception

        target_x0 = np.array([10000.0, 5000.0, 5000.0, -400.0, -200.0, -100.0])
        missile_x0 = np.array([0.0, 0.0, 0.0, 300.0, 200.0, 500.0])

        m_traj, _, _, intercepted, min_miss = simulate_interception(
            missile_x0=missile_x0,
            target_x0=target_x0,
            steps=1000,
            dt=0.05,
            N=4.0,
            kill_radius=15.0
        )

        self.assertTrue(intercepted, f"Interception failed! Min miss distance was {min_miss:.2f} m")
        self.assertLessEqual(min_miss, 15.0)


if __name__ == '__main__':
    unittest.main()
