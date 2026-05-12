import argparse
import pygame
import numpy as np
from track import generate_track
from renderer import Renderer
from path_manager import PathManager, PerfectPathManager
from kart import KartState, KartDynamics
from controller.pid_controller import PIDController
from controller.pid import PID
from logger import logger
from stats import Stats
from sensors.gps import SILGPSSensor
from sensors.hall import SILHallSensor
from sensors.imu import SILIMUSensor
from state_estimation.ekf import EKF

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
    WHEEL_RADIUS = 0.13
    KART_WIDTH = 0.70
    WHEEL_BASE = 1.05

    # Sensors
    NUM_MAGNETS=8
    HALL_UPDATE_RATE = 50
    IMU_UPDATE_RATE = 200
    GPS_UPDATE_RATE = 10

    def make_state(track):
        cl = track.centerline()
        hdg = np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0])
        return KartState(x=cl[0, 0], y=cl[0, 1], heading_rad=hdg, speed_ms=0.0, steer_angle_rad=0)

    track = generate_track(seed=args.seed, num_samples=args.samples, track_width=args.width)
    if not track:
        logger.error("Could not generate track")
        return

    actual_state = make_state(track)
    
    path_manager: PathManager = PerfectPathManager(
        m=MASS_KG, mu=MU_S_COEFF, max_speed_ms=MAX_SPEED_MS,
        max_accel=MAX_ACCEL, max_brake=MAX_BRAKE,
    )

    hall_sensor = SILHallSensor(
        NUM_MAGNETS,
        HALL_UPDATE_RATE,
        WHEEL_RADIUS,
        WHEEL_BASE,
        KART_WIDTH,
    )
    imu_sensor = SILIMUSensor(
        IMU_UPDATE_RATE,
        WHEEL_BASE,
    )
    gps_sensor = SILGPSSensor(
        GPS_UPDATE_RATE
    )

    ekf = EKF()
    ekf.x[0] = actual_state.x
    ekf.x[1] = actual_state.y
    ekf.x[4] = actual_state.heading_rad

    throttle_pid = PID(
        kp=0.5,
        ki=0.2,
        kd=0.1,
        i_awup=2.0,
    )
    lateral_pid = PID(
        kp=0.7,
        ki=0.1,
        kd=0.0,
    )

    controller = PIDController(
        WHEEL_BASE,
        throttle_pid,
        lateral_pid,
    )
    # controller = PurePursuitController()
    dynamics = KartDynamics(
        wheel_base=WHEEL_BASE, max_steer=MAX_STEER_RAD, max_speed=MAX_SPEED_MS,
        max_accel=MAX_ACCEL, max_brake=MAX_BRAKE, drag=DRAG_COEFF,
    )
    stats = Stats(track)

    renderer = Renderer()

    sim_speed = 1.0
    sim_time = 0.0
    paused = False
    physics_dt = 1 / 250
    physics_acc = 0.0
    trail = []

    logger.info("Computing racing line...")
    path_manager.compute_path(actual_state.x, actual_state.y, actual_state.heading_rad, actual_state.speed_ms, track)
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
                        actual_state = make_state(track)
                        path_manager.cached_track_id = None
                        trail.clear()
                        
                        ekf = EKF()
                        ekf.x[0] = actual_state.x
                        ekf.x[1] = actual_state.y
                        ekf.x[4] = actual_state.heading_rad
                elif event.key == pygame.K_EQUALS:
                    sim_speed = min(sim_speed * 2, 16.0)
                elif event.key == pygame.K_MINUS:
                    sim_speed = max(sim_speed / 2, 0.25)

        if not paused:
            physics_acc += frame_dt * sim_speed
            while physics_acc >= physics_dt:
                imu = imu_sensor.measure(actual_state, sim_time)
                gps = gps_sensor.measure(actual_state, sim_time)
                hall = hall_sensor.measure(actual_state, sim_time)

                if imu:
                    ekf.predict(imu, physics_dt)
                if gps:
                    ekf.update_gps(gps)
                if hall:
                    ekf.update_hall(hall)

                fused = ekf.get_state()

                measured_state = KartState(
                    fused.px,
                    fused.py,
                    fused.heading,
                    np.sqrt(fused.vx**2 + fused.vy**2),
                    actual_state.steer_angle_rad
                )

                path = path_manager.compute_path(measured_state.x, measured_state.y, measured_state.heading_rad, measured_state.speed_ms, track)
                throttle, steer = controller.compute(measured_state.x, measured_state.y, measured_state.heading_rad, measured_state.speed_ms, path, physics_dt)
                actual_state = dynamics.step(actual_state, throttle, steer, physics_dt)
                
                physics_acc -= physics_dt
                sim_time += physics_dt
                
                trail.append((actual_state.x, actual_state.y))
                if len(trail) > 5000:
                    trail.pop(0)
                
                stats.iter(actual_state, fused, physics_dt)

        renderer.set_camera(actual_state.x, actual_state.y)
        renderer.clear()
        renderer.draw_track(track)

        if hasattr(path_manager, 'cached_path') and path_manager.cached_path is not None:
            renderer.draw_racing_line(path_manager.cached_path.waypoints)

        renderer.draw_trail(trail)
        renderer.draw_kart(actual_state.x, actual_state.y, actual_state.heading_rad)

        hud = stats.hud_lines(actual_state) + [
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
