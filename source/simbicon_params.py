# All models

IK_GAIN = 0
POSITION_BALANCE_GAIN = 1
VELOCITY_BALANCE_GAIN = 2
TORSO_WORLD = 3
STANCE_ANKLE_RELATIVE = 4

UP_IDX = 5
DN_IDX = 9
# These 4 params are different depending on whether we're going "up" or "down"
SWING_HIP_WORLD = 0
SWING_KNEE_RELATIVE = 1
SWING_ANKLE_RELATIVE = 2
STANCE_KNEE_RELATIVE = 3

# 3D models

POSITION_BALANCE_GAIN_LAT = 13
VELOCITY_BALANCE_GAIN_LAT = 14
HEADING = 15
STANCE_HIP_ROLL_EXTRA = 16
STANCE_ANKLE_ROLL = 17
SWING_ANKLE_ROLL = 18

# Darwin
UP_DURATION = 19

# Add experimental new params at the bottom
STANCE_YAW = 20
TORSO_ROLL = 21

import numpy as np
MIRROR_PARAMS = np.array([1,1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,-1,-1,-1,-1, 1,-1,-1])
PARAM_SCALE   = np.array([1,1,1,1,3, 1,1,1,2, 2,2,3,2, 1,1, 1, 1, 3, 3, 1, 1, 1])

N_PARAMS = len(PARAM_SCALE)
