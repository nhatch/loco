# These should be the same for all 3D models
import numpy as np

BRICK_DOF = 6
Q_DIM = 21

X = 0 # forward
Y = 1 # vertical
Z = 2 # transverse

ROOT_PITCH = 3
ROOT_YAW = 4
ROOT_ROLL = 5

LEG_DOF = 6
RIGHT_IDX = 6
LEFT_IDX = 12

HIP_PITCH = 0
HIP_YAW = 1
HIP_ROLL = 2
KNEE = 3
ANKLE = 4
ANKLE_ROLL = 5

TORSO_ROLL = 18
TORSO_YAW = 20

observable_features_q = np.array([
        1,0,1,1,1,1,
        1,0,1,0,0,0,
        1,0,1,0,0,0,
        0,0,0]) == 1 # Bool array
observable_features_t = np.array([1,0,1]) == 1

absolute_rotation_indices = [Z, ROOT_YAW, ROOT_ROLL,
        RIGHT_IDX + HIP_YAW, RIGHT_IDX + HIP_ROLL, RIGHT_IDX + ANKLE_ROLL,
        LEFT_IDX + HIP_YAW, LEFT_IDX + HIP_ROLL, LEFT_IDX + ANKLE_ROLL,
        TORSO_ROLL, TORSO_YAW]

DEFAULT_GROUND_WIDTH = 2.0
