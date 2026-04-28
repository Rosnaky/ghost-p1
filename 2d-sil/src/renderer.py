import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from track import Track


class TrackRenderer:
    def __init__(self, track: Track):
        self.track = track

    def plot_track(self, racing_line=None, show=True, save_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        self._draw_surface(ax)
        self._draw_boundaries(ax)
        self._draw_centerline(ax)
        self._draw_start_finish(ax)

        if racing_line is not None:
            self._draw_racing_line(ax, racing_line)

        self._draw_curvature_markers(ax)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Track Layout")
        ax.grid(True, alpha=0.3)

        # Legend
        handles = [
            mpatches.Patch(color="grey", alpha=0.3, label="Track surface"),
            plt.Line2D([], [], color="white", linestyle="--", alpha=0.5, label="Centerline"), # type: ignore
            plt.Line2D([], [], color="black", linewidth=2, label="Boundaries"), # type: ignore
        ]
        if racing_line is not None:
            handles.append(plt.Line2D([], [], color="red", linewidth=2, label="Racing line")) # type: ignore
        ax.legend(handles=handles, loc="upper right")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        if show:
            plt.show()
        return fig, ax

    def plot_curvature_profile(self, show=True):
        fig, ax = plt.subplots(figsize=(12, 4))
        s = self.track.points["s"]
        k = self.track.points["curvature"]

        ax.fill_between(s, k, alpha=0.3, color="blue") # type: ignore
        ax.plot(s, k, color="blue", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("Arc length (m)")
        ax.set_ylabel("Curvature (1/m)")
        ax.set_title("Curvature Profile")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def animate_lap(self, racing_line, interval_ms=20, kart_size=1.5):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self._draw_surface(ax)
        self._draw_boundaries(ax)
        # self._draw_racing_line(ax, racing_line)  # remove this
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        kart_dot, = ax.plot([], [], "o", color="red", markersize=8, zorder=10)
        heading_line, = ax.plot([], [], "-", color="red", linewidth=2, zorder=10)
        trail_line, = ax.plot([], [], "-", color="red", alpha=0.4, linewidth=1, zorder=5)

        n = len(racing_line)

        def update(frame):
            i = frame % n
            px, py = racing_line[i, 0], racing_line[i, 1]
            i_next = (i + 1) % n
            hdg = np.arctan2(
                racing_line[i_next, 1] - py,
                racing_line[i_next, 0] - px,
            )
            hx = px + kart_size * np.cos(hdg)
            hy = py + kart_size * np.sin(hdg)

            kart_dot.set_data([px], [py])
            heading_line.set_data([px, hx], [py, hy])
            trail_line.set_data(racing_line[:i+1, 0], racing_line[:i+1, 1])
            return kart_dot, heading_line, trail_line

        anim = FuncAnimation(
            fig, update, frames=n, interval=interval_ms, blit=True,
        )
        plt.show()
        return anim

    # Local helpers

    def _draw_surface(self, ax):
        left = self.track.left_boundary()
        right = self.track.right_boundary()

        # Close the polygon: left forward, right reversed
        poly_x = np.concatenate([left[:, 0], right[::-1, 0]])
        poly_y = np.concatenate([left[:, 1], right[::-1, 1]])
        ax.fill(poly_x, poly_y, color="grey", alpha=0.3, zorder=0)

    def _draw_boundaries(self, ax):
        left = self.track.left_boundary()
        right = self.track.right_boundary()

        # Close the loop for drawing
        for boundary, color in [(left, "black"), (right, "black")]:
            closed = np.vstack([boundary, boundary[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2, zorder=2)

    def _draw_centerline(self, ax):
        center = self.track.centerline()
        closed = np.vstack([center, center[0]])
        ax.plot(closed[:, 0], closed[:, 1], "w--", alpha=0.5, linewidth=1, zorder=1)

    def _draw_start_finish(self, ax):
        p = self.track.points
        lx, ly = p["left_x"][0], p["left_y"][0]
        rx, ry = p["right_x"][0], p["right_y"][0]
        ax.plot([lx, rx], [ly, ry], color="green", linewidth=3, zorder=5)
        ax.annotate(
            "S/F", xy=((lx + rx) / 2, (ly + ry) / 2),
            fontsize=10, fontweight="bold", color="green",
            ha="center", va="bottom", zorder=5,
        )

    def _draw_racing_line(self, ax, racing_line):
        if racing_line.shape[1] >= 3:
            # Color by speed
            points = racing_line[:, :2].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            speeds = racing_line[:-1, 2]
            lc = LineCollection(segments, cmap="RdYlGn", linewidth=2.5, zorder=4) # type: ignore
            lc.set_array(speeds)
            ax.add_collection(lc)
            plt.colorbar(lc, ax=ax, label="Speed (m/s)", shrink=0.6)
        else:
            closed = np.vstack([racing_line[:, :2], racing_line[0, :2]])
            ax.plot(closed[:, 0], closed[:, 1], color="red", linewidth=2, zorder=4)

    def _draw_curvature_markers(self, ax):
        p = self.track.points
        k = np.abs(p["curvature"])
        threshold = np.percentile(k, 85)  # mark the sharpest 15% of turns
        mask = k > threshold

        ax.quiver(
            p["x"][mask], p["y"][mask],
            np.cos(p["heading"][mask]), np.sin(p["heading"][mask]),
            angles="xy", scale=50, width=0.003,
            color="orange", alpha=0.6, zorder=3,
        )
