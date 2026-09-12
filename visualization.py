import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from environment import simulate_interception


def animate_trajectory(trajectory, target_trajectory=None, interval=30, title='Patriot Engine: Proximity Scaling & Dynamic Glow'):
    current_trajectory = trajectory
    current_target_trajectory = target_trajectory
    steps = int(current_trajectory.shape[0])
    frame_state = {"frame": 0}

    fig = plt.figure(figsize=(8, 8), facecolor='#111111')
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    ax.set_facecolor('#111111')

    # Mouse drag: make view follow cursor movement (use azim -= dx, elev += dy)
    _drag = {"x": None, "y": None, "pressed": False}

    def _on_press(event):
        if event.inaxes != ax:
            return
        # ignore if toolbar is in any interactive mode (pan/zoom/etc.)
        def _toolbar_mode():
            tb = getattr(fig.canvas, 'toolbar', None)
            if tb and getattr(tb, 'mode', None):
                return getattr(tb, 'mode')
            mgr = getattr(fig.canvas, 'manager', None)
            if mgr:
                tb2 = getattr(mgr, 'toolbar', None)
                if tb2 and getattr(tb2, 'mode', None):
                    return getattr(tb2, 'mode')
            return ''

        if _toolbar_mode():
            return
        # only start custom drag on left mouse button
        if getattr(event, 'button', 1) != 1:
            return
        _drag['pressed'] = True
        _drag['x'] = event.x
        _drag['y'] = event.y

    def _on_release(event):
        _drag['pressed'] = False

    def _on_motion(event):
        if not _drag['pressed'] or event.inaxes != ax:
            return
        # respect toolbar interactive mode (any non-empty mode)
        def _toolbar_mode_check():
            tb = getattr(fig.canvas, 'toolbar', None)
            if tb and getattr(tb, 'mode', None):
                return True
            mgr = getattr(fig.canvas, 'manager', None)
            if mgr:
                tb2 = getattr(mgr, 'toolbar', None)
                if tb2 and getattr(tb2, 'mode', None):
                    return True
            return False

        if _toolbar_mode_check():
            return
        dx = event.x - (_drag['x'] or event.x)
        dy = event.y - (_drag['y'] or event.y)
        sens_x = 0.10  # slower horizontal movement
        sens_y = 0.25  # vertical sensitivity
        azim = ax.azim + dx * sens_x
        elev = ax.elev + dy * sens_y
        ax.view_init(elev, azim)
        _drag['x'] = event.x
        _drag['y'] = event.y
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)

    def _update_bounds(m_traj, t_traj):
        all_pts = m_traj[:, :3]
        if t_traj is not None:
            all_pts = np.vstack([all_pts, t_traj[:, :3]])

        min_x, max_x = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        min_y, max_y = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
        min_z, max_z = np.min(all_pts[:, 2]), np.max(all_pts[:, 2])

        pad_x = (max_x - min_x) * 0.08 if max_x != min_x else 1000.0
        pad_y = (max_y - min_y) * 0.08 if max_y != min_y else 1000.0
        pad_z = (max_z - min_z) * 0.08 if max_z != min_z else 1000.0

        ax.set_xlim([min_x - pad_x, max_x + pad_x])
        ax.set_ylim([min_y - pad_y, max_y + pad_y])
        ax.set_zlim([0, max_z + pad_z])
        ax.set_box_aspect([1, 1, 1])

    _update_bounds(current_trajectory, current_target_trajectory)

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
    ax.set_title(title, color='white', fontsize=12)
    ax.grid(True, color='#333333', linestyle='--')

    trail_line, = ax.plot([], [], [], ':', color='cyan', linewidth=1.0, alpha=0.3, label='Flight Path')
    missile_head, = ax.plot([], [], [], 'o', color='red', markersize=4, alpha=0.3, label='Missile (State x)')
    # target artist (if provided)
    target_head = None
    target_trail = None
    if target_trajectory is not None:
        target_trail, = ax.plot([], [], [], ':', color='yellow', linewidth=1.0, alpha=0.6, label='Target Path')
        target_head, = ax.plot([], [], [], 'X', color='yellow', markersize=8, alpha=0.9, label='Target')

    ax.legend(loc='upper right', facecolor='#222222', edgecolor='white', fontsize=9)

    stats_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=9, color='white',
                           va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222", edgecolor="cyan", alpha=0.9))
    target_stats_text = ax.text2D(0.02, 0.82, "", transform=ax.transAxes, fontsize=9, color='white',
                                  va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222", edgecolor="yellow", alpha=0.9))

    restart_ax = fig.add_axes([0.82, 0.02, 0.14, 0.05])
    restart_button = Button(restart_ax, 'Restart', color='#222222', hovercolor='#333333')
    restart_button.label.set_color('white')

    def _restart_simulation(_event):
        nonlocal current_trajectory, current_target_trajectory, steps
        m_traj, t_traj, _, intercepted, min_miss = simulate_interception()
        current_trajectory = m_traj
        current_target_trajectory = t_traj
        steps = int(current_trajectory.shape[0])
        _update_bounds(current_trajectory, current_target_trajectory)
        ax.set_title(f"Patriot SAM: ProNav Interception (Miss Distance: {min_miss:.1f} m)", color='white', fontsize=12)
        frame_state["frame"] = 0
        ani.frame_seq = ani.new_frame_seq()
        fig.canvas.draw_idle()

    restart_button.on_clicked(_restart_simulation)

    def animate(frame):
        frame_state["frame"] = frame

        marker_size = 8.0
        line_width = 1.5
        alpha_val = 0.9

        missile_color = (1.0, 0.1, 0.0, alpha_val)
        trail_color = (0.0, 0.8, 1.0, alpha_val)

        trail_line.set_data(current_trajectory[:frame+1, 0], current_trajectory[:frame+1, 1])
        trail_line.set_3d_properties(current_trajectory[:frame+1, 2])
        trail_line.set_linewidth(line_width)
        trail_line.set_color(trail_color)

        missile_head.set_data([current_trajectory[frame, 0]], [current_trajectory[frame, 1]])
        missile_head.set_3d_properties([current_trajectory[frame, 2]])
        missile_head.set_markersize(marker_size)
        missile_head.set_color(missile_color)

        # if target trajectory provided, update its artists
        if current_target_trajectory is not None and target_head is not None:
            target_head.set_data([current_target_trajectory[frame, 0]], [current_target_trajectory[frame, 1]])
            target_head.set_3d_properties([current_target_trajectory[frame, 2]])
        if current_target_trajectory is not None and target_trail is not None:
            target_trail.set_data(current_target_trajectory[:frame+1, 0], current_target_trajectory[:frame+1, 1])
            target_trail.set_3d_properties(current_target_trajectory[:frame+1, 2])

        pos = current_trajectory[frame, :3]
        vel = current_trajectory[frame, 3:]
        speed = np.linalg.norm(vel)
        stats_text.set_text(
            f"Step k = {frame}/{steps}\n"
            f"Pos (m): [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]\n"
            f"Speed: {speed:.1f} m/s\n"
        )

        if current_target_trajectory is not None:
            target_pos = current_target_trajectory[frame, :3]
            target_vel = current_target_trajectory[frame, 3:]
            target_speed = np.linalg.norm(target_vel)
            target_range = np.linalg.norm(target_pos)
            target_stats_text.set_text(
                f"Target telemetry\n"
                f"Pos (m): [{target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f}]\n"
                f"Speed: {target_speed:.1f} m/s\n"
                f"Range: {target_range:.0f} m\n"
            )
        else:
            target_stats_text.set_text("")

        if current_target_trajectory is not None and target_head is not None and target_trail is not None:
            return trail_line, missile_head, target_trail, target_head, stats_text, target_stats_text
        if current_target_trajectory is not None and target_head is not None:
            return trail_line, missile_head, target_head, stats_text, target_stats_text
        return trail_line, missile_head, stats_text, target_stats_text

    ani = FuncAnimation(fig, animate, frames=steps, interval=interval, blit=False)
    plt.show()
    return ani
