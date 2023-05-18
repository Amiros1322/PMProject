import numpy as np
from tqdm import tqdm

from playground.bguav.vehicle_state_estimator.src.runge_kutta4 import propagate_state
from playground.bguav.vehicle_state_estimator.src.motion_models.kbm import KinematicBicycleModel



def test_kbm():
    dt = 0.1
    x0 = np.array([0, 0, 0, 0, 1, 0, 0]).reshape(-1, 1)
    delta = np.deg2rad(5)
    kbm = KinematicBicycleModel(L=1.5)
    x = propagate_state(f=kbm.f, dt=dt, x=x0, u=delta)
    print(x)


if __name__ == "__main__":
    test_kbm()