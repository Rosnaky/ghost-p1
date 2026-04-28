
import argparse
from track import Track, generate_track
from renderer import TrackRenderer
from logger import logger
from path_manager import PathManager, CenterlineConstantSpeedPathManager, PerfectPathManager
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
    
    MAX_SPEED_MS = 60
    MAX_ACCEL = 25
    MAX_BRAKE = 15
    DRAG_COEFF = 0.35
    MAX_STEER_RAD = np.radians(25)
    MU_S_COEFF = 0.6
    MASS_KG = 60

    path_manager: PathManager = PerfectPathManager(
        m=MASS_KG,
        mu=MU_S_COEFF, 
        max_speed_ms=MAX_SPEED_MS,
        max_accel=MAX_ACCEL,
        max_brake=MAX_BRAKE
    )
    # path_manager: PathManager = CenterlineConstantSpeedPathManager()
    controller = PurePursuitController()

    dynamics = KartDynamics(
        wheel_base=1.05, 
        max_steer=MAX_STEER_RAD, 
        max_speed=MAX_SPEED_MS,
        max_accel=MAX_ACCEL,
        max_brake=MAX_BRAKE,
        drag=DRAG_COEFF
    )
    dt = 1 / 20 # 100 Hz

    positions = []

    for i in range(5000):
        path = path_manager.compute_path(state.x, state.y, state.heading_rad, state.speed_ms, track)
        throttle, steer = controller.compute(state.x, state.y, state.heading_rad, state.speed_ms, path)
        state = dynamics.step(state, throttle, steer, dt)
        
        positions.append([state.x, state.y])

        logger.debug(f"Iteration {i} complete")

    positions = np.array(positions)
    renderer.animate_lap(positions)

if __name__ == "__main__":
    main()
