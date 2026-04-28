
import argparse
from track import Track, generate_track
from renderer import TrackRenderer
from logger import logger
from path_manager import PathManager
from path_manager import CenterlineConstantSpeedPathManager
from kart import KartState, KartDynamics
from controller.pure_pursuit_controller import PurePursuitController
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(description="Ghost-P1 2D SIL")
    parser.add_argument("--seed", type=int, default=69, help="Track generation seed")
    parser.add_argument("--samples", type=int, default=500, help="Number of track sample points")
    parser.add_argument("--width", type=float, default=5.0, help="Track width in meters")
    args = parser.parse_args()

    track = generate_track(seed=args.seed, num_samples=args.samples, track_width=args.width)

    if not track:
        logger.error("Could not generate track")
        return

    logger.info(f"Track generated: {track.total_length:.1f}m, {track.num_points} points")

    cl = track.centerline()
    heading = np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0])
    state = KartState(
        x=cl[0, 0], 
        y=cl[0, 1], 
        heading_rad=heading,
        speed_ms=0.0,
        steer_angle_rad=0,
    )

    renderer = TrackRenderer(track)
    
    path_manager: PathManager = CenterlineConstantSpeedPathManager()
    controller = PurePursuitController()

    dynamics = KartDynamics()
    dt = 1 / 100 # 100 Hz

    positions = []

    for _ in range(5000):
        path = path_manager.compute_path(state.x, state.y, state.heading_rad, state.speed_ms, track)
        throttle, steer = controller.compute(state.x, state.y, state.heading_rad, state.speed_ms, path)
        state = dynamics.step(state, throttle, steer, dt)
        
        positions.append([state.x, state.y])

    positions = np.array(positions)
    renderer.animate_lap(positions)

if __name__ == "__main__":
    main()
