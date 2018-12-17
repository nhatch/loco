import numpy as np

perm = [0,1,2,3,4,5,6,7,8]
sign_switches = []

BRICK_DOF = 3
CONTROL_BOUNDS = 1.5 * np.array([100, 100, 20, 100, 100, 20])

Q_DIM = 9
X = 0
Y= 1
ROOT_PITCH = 2

LEG_DOF = 3
RIGHT_IDX = 3
LEFT_IDX = 6

HIP_PITCH = 0
KNEE = 1
ANKLE = 2

PELVIS_BODYNODE_IDX = 2
RIGHT_BODYNODE_IDX = 3
LEFT_BODYNODE_IDX = 6
THIGH_BODYNODE_OFFSET = 0
FOOT_BODYNODE_OFFSET = 2

L_THIGH =    0.45
L_SHIN =   0.50
L_FOOT =   0.20
FOOT_RADIUS = 0.06

KP_GAIN = [200.0]*6
KD_GAIN = [15.0]*6

ALLOWED_COLLISION_IDS = [5,8] # The two feet

observable_features_q = np.ones(Q_DIM) == 1 # Bool array
observable_features_t = np.array([1,1,0]) == 1
