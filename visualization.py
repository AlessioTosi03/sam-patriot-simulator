import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_trajectory(trajectory, target_trajectory=None, interval=30, title='Patriot Engine: Proximity Scaling & Dynamic Glow'):
    steps = int(trajectory.shape[0])

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

    max_range = np.max(trajectory[:, :3]) * 1.05
    if max_range == 0:
        max_range = 1.0
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])
    ax.set_box_aspect([1, 1, 1])

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

    def animate(frame):
        progress = frame / max(1, steps - 1)

        marker_size = 8.0
        line_width = 1.5
        alpha_val = 0.9

        missile_color = (1.0, 0.1, 0.0, alpha_val)
        trail_color = (0.0, 0.8, 1.0, alpha_val)

        trail_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        trail_line.set_3d_properties(trajectory[:frame+1, 2])
        trail_line.set_linewidth(line_width)
        trail_line.set_color(trail_color)

        missile_head.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        missile_head.set_3d_properties([trajectory[frame, 2]])
        missile_head.set_markersize(marker_size)
        missile_head.set_color(missile_color)

        # if target trajectory provided, update its artists
        if target_trajectory is not None and target_head is not None:
            target_head.set_data([target_trajectory[frame, 0]], [target_trajectory[frame, 1]])
            target_head.set_3d_properties([target_trajectory[frame, 2]])
        if target_trajectory is not None and target_trail is not None:
            target_trail.set_data(target_trajectory[:frame+1, 0], target_trajectory[:frame+1, 1])
            target_trail.set_3d_properties(target_trajectory[:frame+1, 2])

        pos = trajectory[frame, :3]
        vel = trajectory[frame, 3:]
        speed = np.linalg.norm(vel)
        stats_text.set_text(
            f"Step k = {frame}/{steps}\n"
            f"Pos (m): [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]\n"
            f"Speed: {speed:.1f} m/s\n"
        )

        if target_head is not None and target_trail is not None:
            return trail_line, missile_head, target_trail, target_head, stats_text
        if target_head is not None:
            return trail_line, missile_head, target_head, stats_text
        return trail_line, missile_head, stats_text

    ani = FuncAnimation(fig, animate, frames=steps, interval=interval, blit=False)
    plt.show()
    return ani
