import numpy as np
from dataclasses import dataclass


@dataclass
class LOSState:
    """
    Data class representing the Line-of-Sight (LOS) kinematic state
    between the interceptor missile and the target threat.
    """
    r_vec: np.ndarray        # Relative position vector [Rx, Ry, Rz] (m)
    range: float             # Relative distance magnitude R (m)
    u_r: np.ndarray          # Unit LOS vector u_r = R / ||R||
    v_rel: np.ndarray        # Relative velocity vector V_rel = V_target - V_missile (m/s)
    v_closing: float         # Closing velocity V_c = - (R . V_rel) / R (m/s)
    omega_los: np.ndarray    # LOS angular rate vector Omega = (R x V_rel) / R^2 (rad/s)


def compute_los(target_state: np.ndarray, missile_state: np.ndarray) -> LOSState:
    """
    Computes the 3D Line-of-Sight (LOS) kinematic state between missile and target.
    
    Parameters:
        target_state:  np.ndarray of shape (6,) [x, y, z, vx, vy, vz]
        missile_state: np.ndarray of shape (6,) [x, y, z, vx, vy, vz]
        
    Returns:
        LOSState dataclass containing calculated relative kinematics.
    """
    r_vec = target_state[:3] - missile_state[:3]
    range_mag = float(np.linalg.norm(r_vec))
    
    # Avoid division by zero when intercepting (range_mag -> 0)
    eps = 1e-6
    if range_mag < eps:
        u_r = np.zeros(3)
        range_mag = 0.0
    else:
        u_r = r_vec / range_mag
        
    v_rel = target_state[3:] - missile_state[3:]
    
    if range_mag > eps:
        v_closing = -float(np.dot(r_vec, v_rel) / range_mag)
        omega_los = np.cross(r_vec, v_rel) / (range_mag ** 2)
    else:
        v_closing = 0.0
        omega_los = np.zeros(3)
        
    return LOSState(
        r_vec=r_vec,
        range=range_mag,
        u_r=u_r,
        v_rel=v_rel,
        v_closing=v_closing,
        omega_los=omega_los
    )


def compute_pronav_acceleration(
    target_state: np.ndarray,
    missile_state: np.ndarray,
    N: float = 3.5,
    max_accel: float = 300.0,
    include_gravity: bool = True
) -> tuple[np.ndarray, LOSState]:
    """
    Computes the 3D Proportional Navigation (ProNav) acceleration command u(k) = a_M(k).
    
    Mathematical Law:
        a_PRONAV = N * V_c * (Omega_LOS x u_R)
        
    Parameters:
        target_state:  Target state vector [x, y, z, vx, vy, vz]
        missile_state: Missile state vector [x, y, z, vx, vy, vz]
        N:             Navigation constant (typically 3.0 to 5.0)
        max_accel:     Maximum structural acceleration maneuver limit in m/s^2 (e.g. 30 G ~ 300 m/s^2)
        include_gravity: If True, adds gravity compensation (+g on z) to maintain trim flight
        
    Returns:
        (accel_cmd, los_state):
            accel_cmd: 3D acceleration vector [ax, ay, az] in m/s^2
            los_state: LOSState dataclass
    """
    los = compute_los(target_state, missile_state)
    
    # ProNav Guidance Law: a_cmd = N * V_c * (Omega x u_r)
    # Note: ProNav requires positive closing velocity (V_c > 0)
    if los.v_closing > 0.0 and los.range > 1.0:
        a_cmd = N * los.v_closing * np.cross(los.omega_los, los.u_r)
    else:
        a_cmd = np.zeros(3)
        
    # Optional gravity bias compensation so missile doesn't drop due to g=9.81
    if include_gravity:
        a_cmd[2] += 9.81
        
    # Apply physical structural acceleration saturation limit (G-limit)
    accel_mag = np.linalg.norm(a_cmd)
    if accel_mag > max_accel:
        a_cmd = a_cmd * (max_accel / accel_mag)
        
    return a_cmd, los
