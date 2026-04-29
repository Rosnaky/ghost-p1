import argparse
import pygame
import numpy as np
from track import generate_track
from renderer import Renderer
from path_manager import PathManager, CenterlineConstantSpeedPathManager, PerfectPathManager
from kart import KartState, KartDynamics
from controller.pure_pursuit_controller import PurePursuitController
from logger import logger
from stats import Stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--width", type=float, default=5.0)
    args = parser.parse_args()

    MAX_SPEED_MS = 60
    MAX_ACCEL = 10
    MAX_BRAKE = 5
    DRAG_COEFF = 0.30
    MAX_STEER_RAD = np.radians(45)
    MU_S_COEFF = 0.7
    MASS_KG = 60

    def make_state(track):
        cl = track.centerline()
        hdg = np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0])
        return KartState(x=cl[0, 0], y=cl[0, 1], heading_rad=hdg, speed_ms=0.0, steer_angle_rad=0)

    track = generate_track(seed=args.seed, num_samples=args.samples, track_width=args.width)
    if not track:
        logger.error("Could not generate track")
        return

    state = make_state(track)

    path_manager: PathManager = PerfectPathManager(
        m=MASS_KG, mu=MU_S_COEFF, max_speed_ms=MAX_SPEED_MS,
        max_accel=MAX_ACCEL, max_brake=MAX_BRAKE,
    )
    controller = PurePursuitController()
    dynamics = KartDynamics(
        wheel_base=1.05, max_steer=MAX_STEER_RAD, max_speed=MAX_SPEED_MS,
        max_accel=MAX_ACCEL, max_brake=MAX_BRAKE, drag=DRAG_COEFF,
    )
    stats = Stats(track)

    renderer = Renderer()

    sim_speed = 1.0
    paused = False
    physics_dt = 1 / 20
    physics_acc = 0.0
    trail = []

    logger.info("Computing racing line...")
    path_manager.compute_path(state.x, state.y, state.heading_rad, state.speed_ms, track)
    logger.info("Done.")

    running = True
    while running:
        frame_dt = renderer.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                renderer.handle_scroll(event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    seed = np.random.randint(0, 100000)
                    new_track = generate_track(seed=seed, num_samples=args.samples, track_width=args.width)
                    if new_track:
                        track = new_track
                        state = make_state(track)
                        path_manager.cached_track_id = None
                        trail.clear()
                elif event.key == pygame.K_EQUALS:
                    sim_speed = min(sim_speed * 2, 16.0)
                elif event.key == pygame.K_MINUS:
                    sim_speed = max(sim_speed / 2, 0.25)

        if not paused:
            physics_acc += frame_dt * sim_speed
            while physics_acc >= physics_dt:
                path = path_manager.compute_path(state.x, state.y, state.heading_rad, state.speed_ms, track)
                throttle, steer = controller.compute(state.x, state.y, state.heading_rad, state.speed_ms, path)
                state = dynamics.step(state, throttle, steer, physics_dt)
                physics_acc -= physics_dt
                trail.append((state.x, state.y))
                if len(trail) > 5000:
                    trail.pop(0)
                
                stats.iter(state, physics_dt)

        renderer.set_camera(state.x, state.y)
        renderer.clear()
        renderer.draw_track(track)

        if hasattr(path_manager, 'cached_path') and path_manager.cached_path is not None:
            renderer.draw_racing_line(path_manager.cached_path.waypoints)

        renderer.draw_trail(trail)
        renderer.draw_kart(state.x, state.y, state.heading_rad)

        hud = stats.hud_lines(state) + [
            f"Zoom: {renderer.zoom:.1f}x  Sim: {sim_speed:.1f}x",
            f"Path: {path_manager.name}",
            f"{'PAUSED' if paused else 'RUNNING'}",
            f"[R] new track  [SPACE] pause  [+/-] speed  [scroll] zoom",
        ]

        renderer.draw_hud(hud)

        renderer.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
