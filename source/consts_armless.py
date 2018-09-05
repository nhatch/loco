import numpy as np
skel_file = "skel/HumanSkel/kima_human_box_armless.skel"

BRICK_DOF = 6
CONTROL_BOUNDS = 1000 * np.array([100]*31)

LEG_DOF = 6
RIGHT_IDX = 12
LEFT_IDX = 6
Q_DIM = 21
PITCH_IDX = 3
YAW_IDX = 4
ROLL_IDX = 5
X_IDX = 2
Y_IDX = 1

HIP_OFFSET = 0
HIP_OFFSET_TWIST = 1
HIP_OFFSET_LAT = 2 #Backward sign
KNEE_OFFSET = 3
ANKLE_OFFSET = 4
ANKLE_OFFSET_LAT = 5

PELVIS_BODYNODE_IDX = 1
RIGHT_BODYNODE_IDX = 7 # Used for finding contacts and locating the heel
LEFT_BODYNODE_IDX = 4

# TODO this skel file has a lot of "transformation" values that might invalidate
# the current IK code.
L_PELVIS = 0.1855
L_THIGH =    0.42875
L_SHIN =   0.400875
L_FOOT =   0.21
FOOT_RADIUS = 0.049 # Not really a radius; it's a box....

# Three hip, one knee, two ankle
leg_kp = [200, 200, 200, 200, 200, 100]
leg_kd = [15, 15, 15, 15, 15, 10]
# Two abdomen, one chest
trunk_kp = [300, 300, 300]
trunk_kd = [20, 20, 20]
KP_GAIN = leg_kp + leg_kp + trunk_kp
KD_GAIN = leg_kd + leg_kd + trunk_kd

ALLOWED_COLLISION_IDS = [4,7] # The two feet and toes
