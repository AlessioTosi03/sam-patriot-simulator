import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. SIMULATION PARAMETERS
dt = 0.05        # Time step (seconds)
steps = 100      # Total simulation steps

# 2. STATE-SPACE FORMULATION (6D State: [px, py, pz, vx, vy, vz]^T)
A = np.eye(6)
A[0, 3] = dt  # px update
A[1, 4] = dt  # py update
A[2, 5] = dt  # pz update

# Initial State x(0): Launch from origin (0,0,0) with velocity vector (200, 300, 400) m/s
x_0 = np.array([0.0, 0.0, 0.0, 200.0, 300.0, 400.0])

# 3. GENERATE TRAJECTORY VIA DISCRETE TIME SYSTEM: x(k+1) = A * x(k)
trajectory = np.zeros((steps, 6))
x_current = x_0

for k in range(steps):
    trajectory[k] = x_current
    x_current = A @ x_current

# 4. 3D VISUALIZATION SETUP (EQUAL CUBE ASPECT RATIO)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

# Uniform limit across all 3 axes so scale is 1:1:1
max_range = np.max(trajectory[:, :3]) * 1.05

ax.set_xlim([0, max_range])
ax.set_ylim([0, max_range])
ax.set_zlim([0, max_range])

# FORCE THE BOUNDING BOX ASPECT RATIO TO A PERFECT CUBE
ax.set_box_aspect([1, 1, 1])

# Graphic elements
trail_line, = ax.plot([], [], [], 'b--', alpha=0.5, label='Flight Path')
missile_head, = ax.plot([], [], [], 'ro', markersize=8, label='Missile (State x)')

# Axis Labels
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Altitude Z (m)')
ax.set_title('Patriot Engine: 3D Equal-Aspect Cube Space')
ax.legend(loc='upper left')
ax.grid(True)

# Telemetry readout
stats_text = ax.text2D(0.02, 0.85, "", transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

def animate(frame):
    trail_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
    trail_line.set_3d_properties(trajectory[:frame, 2])
    
    missile_head.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
    missile_head.set_3d_properties([trajectory[frame, 2]])
    
    pos = trajectory[frame, :3]
    vel = trajectory[frame, 3:]
    speed = np.linalg.norm(vel)
    stats_text.set_text(
        f"Step k = {frame}\n"
        f"Pos (m): [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]\n"
        f"Speed: {speed:.1f} m/s"
    )
    return trail_line, missile_head, stats_text

ani = FuncAnimation(fig, animate, frames=steps, interval=30, blit=False)
plt.show()