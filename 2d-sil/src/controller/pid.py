import numpy as np

class PID:
    def __init__(self, kp, ki, kd, i_awup=None, output_min=None, output_max=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_awup = i_awup
        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0
        self.prev_error = 0
        self.prev_measurement = 0

    def compute(self, measured, target, dt):
        error = target - measured

        self.integral += error * dt

        if self.i_awup:
            self.integral = np.clip(self.integral, -self.i_awup, self.i_awup)

        derivative = (measured - self.prev_measurement) / (dt + 1e-9)
        self.prev_error = error
        self.prev_measurement = measured

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        sat_min = self.output_min if self.output_min else float("-inf")
        sat_max = self.output_max if self.output_max else float("inf")
        return np.clip(output, sat_min, sat_max)

    def reset(self):
        self.prev_error = 0
        self.integral = 0
