from Tracker.ut.simulation_tests import horizontal_track
from Sensors.Lidar.ut.lidarUtRecord import start_simulator_lidar
from Sensors.IMU.ut.kalman_filter_simulator import start_kalman_filter_simulator
from Sensors.Camera.ut.cameraUtRecord import start_camera_simulator


if __name__ == "__main__":
    print("inside")
    # start_simulator_lidar()

    start_kalman_filter_simulator()
    # start_camera_simulator()
    # horizontal_track()
    