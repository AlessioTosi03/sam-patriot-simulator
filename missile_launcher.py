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

# Initial State x(0): Launch from origin (0,0,0)
x_0 = np.array([0.0, 0.0, 0.0, 200.0, 300.0, 400.0])

# 3. GENERATE TRAJECTORY VIA DISCRETE TIME SYSTEM: x(k+1) = A * x(k)
trajectory = np.zeros((steps, 6))
x_current = x_0

for k in range(steps):
    trajectory[k] = x_current
    x_current = A @ x_current

# 4. 3D VISUALIZATION SETUP (DARK THEME FOR HIGH GLOW/BRIGHTNESS)
fig = plt.figure(figsize=(8, 8), facecolor='#111111')
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.set_facecolor('#111111')

# Set equal cube boundaries
max_range = np.max(trajectory[:, :3]) * 1.05
ax.set_xlim([0, max_range])
ax.set_ylim([0, max_range])
ax.set_zlim([0, max_range])
ax.set_box_aspect([1, 1, 1])

# Dark theme styling for axes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(colors='white')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Altitude Z (m)')
ax.set_title('Patriot Engine: Proximity Scaling & Dynamic Glow', color='white', fontsize=12)
ax.grid(True, color='#333333', linestyle='--')

# Initialize flight objects
trail_line, = ax.plot([], [], [], ':', color='cyan', linewidth=1.0, alpha=0.3, label='Flight Path')
missile_head, = ax.plot([], [], [], 'o', color='red', markersize=4, alpha=0.3, label='Missile (State x)')

# --- GRAPHICAL BUG FIX: SEPARATED CORNERS ---
# 1. Legend placed cleanly in the UPPER RIGHT
ax.legend(loc='upper right', facecolor='#222222', edgecolor='white', labelcolor='white', fontsize=9)

# 2. Telemetry box pinned to UPPER LEFT with top-vertical alignment
stats_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=9, color='white',
                       va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222", edgecolor="cyan", alpha=0.9))

# 5. ANIMATION LOOP WITH DYNAMIC PROXIMITY SCALING
def animate(frame):
    progress = frame / max(1, steps - 1)
    
    # Dynamic properties based on proximity
    marker_size = 4.0 + 16.0 * progress      # Grows from 4px to 20px
    line_width = 1.0 + 3.5 * progress        # Thickens from 1.0 to 4.5
    alpha_val = 0.2 + 0.8 * progress         # Brightens from 0.2 (faint) to 1.0 (vivid)
    
    # RGBA color tuples: Transition to bright neon colors as it approaches
    missile_color = (1.0, 0.1, 0.0, alpha_val)  # Intense glowing neon red
    trail_color = (0.0, 0.8, 1.0, alpha_val)    # Vivid glowing cyan
    
    # Update dotted trail
    trail_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    trail_line.set_3d_properties(trajectory[:frame+1, 2])
    trail_line.set_linewidth(line_width)
    trail_line.set_color(trail_color)
    
    # Update missile head
    missile_head.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
    missile_head.set_3d_properties([trajectory[frame, 2]])
    missile_head.set_markersize(marker_size)
    missile_head.set_color(missile_color)
    
    # Update Telemetry Readout
    pos = trajectory[frame, :3]
    vel = trajectory[frame, 3:]
    speed = np.linalg.norm(vel)
    stats_text.set_text(
        f"Step k = {frame}/{steps}\n"
        f"Pos (m): [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]\n"
        f"Speed: {speed:.1f} m/s\n"
        f"Visual Scale: {marker_size:.1f}px ({progress*100:.0f}%)"
    )
    
    return trail_line, missile_head, stats_text

ani = FuncAnimation(fig, animate, frames=steps, interval=30, blit=False)
plt.show()