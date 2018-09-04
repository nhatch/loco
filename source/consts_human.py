import numpy as np
skel_file = "skel/humanskelmeshes/kima_human_mesh.skel"

BRICK_DOF = 6
CONTROL_BOUNDS = 1000 * np.array([100]*31)

LEG_DOF = 7
RIGHT_IDX = 13
LEFT_IDX = 6
Q_DIM = 37
PITCH_IDX = 2
YAW_IDX = 1
ROLL_IDX = 0
X_IDX = 3
Y_IDX = 4

HIP_OFFSET = 0
HIP_OFFSET_TWIST = 1
HIP_OFFSET_LAT = 2
KNEE_OFFSET = 3
ANKLE_OFFSET = 4
# There is no ankle lat DOF
#ANKLE_OFFSET_LAT = 5
ANKLE_OFFSET_TWIST = 5
# Offset 6 is for the toes

PELVIS_BODYNODE_IDX = 0
RIGHT_BODYNODE_IDX = 7 # Used for finding contacts and locating the heel
LEFT_BODYNODE_IDX = 3

L_PELVIS = 0.1855
L_THIGH =    0.42875
L_SHIN =   0.40075
L_FOOT =   0.21
FOOT_RADIUS = 0.06 # Not really a radius; it's a box....

# I tuned these gains with a single constant target angle, and in zero g.
# Probably there is something fundamentally wrong with doing it this way.
# Three hip, one knee, two ankle, one toe
leg_kp = [200, 200, 200, 200, 200, 200, 8] # Toes need very small gains
leg_kd = [15, 15, 15, 15, 15, 15, 1]
# TODO when tuning these gains I had very strange behavior. For many joints, the robot
# was actually unable to achieve the target angle (even when the angle was within limits).
# E.g. abdomen 1 (index 20), clavicle (index 25), shoulder 1 (index 26)
# Is this due to some kind of self-collision?
# One clavicle (??), 3 shoulder, 1 elbow, 1 wrist
arm_kp = [0, 300, 200, 200, 200, 8]
arm_kd = [0, 20, 15, 15, 15, 1]
# Two abdomen, one spine, two neck
trunk_kp = [300, 300, 300, 200, 300]
trunk_kd = [20, 20, 20, 15, 20]
KP_GAIN = leg_kp + leg_kp + trunk_kp + arm_kp + arm_kp
KD_GAIN = leg_kd + leg_kd + trunk_kd + arm_kd + arm_kd

ALLOWED_COLLISION_IDS = [3,4,7,8] # The two feet and toes
