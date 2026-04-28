# Architecture

The following document highlights the objectives of this project.

## Functionalities
1) Calculates the optimal drive line
2) Calculates the optimal breaking and acceleration pattern
3) Must be able to run on any track without any hardcoding via ML techniques
4) Display the data to the user through either a screen or AR interface

## Inputs
1) Position via GPS (RTK) and Camera
2) Speed via Hall Sensor and GPS (RTK)
3) Acceleration via IMU
4) Heading via Compass

Will need to do sensor fusion + EKF

## Roadmap
### Calculations
1) Create a 2D SIL that will help with the ML portion assuming perfect position and speed estimation
    1) PyGame
2) Create a 3D SIL for computer vision
    1) Panda3D or PyOpenGL

### Hardware
1) GPS RTK
2) EKF
3) Hall Sensor
4) IMU
5) Compass
6) Power/communications
