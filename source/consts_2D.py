import numpy as np
import pydart2.utils.transformations as libtransform
from utils import build_mask

GRAVITY_Y = True
def convert_root(q):
    return q
def inverse_convert_root(q):
    return q

skel_file = 'skel/walker2d.skel'
SIMULATION_FREQUENCY = 2000 # Hz
CONTROL_FREQUENCY = 2000
REAL_TIME_STEPS_PER_RENDER = 25

perm = [0,1,2,3,4,5,6,7,8]
sign_switches = []

BRICK_DOF = 3
cb = 1.5 * np.array([100, 100, 20, 100, 100, 20])
CONTROL_BOUNDS = np.array([-cb, cb])

Q_DIM = 9
Q_DIM_RAW = 9
X = 0
Y= 1
ROOT_PITCH = 2

LEG_DOF = 3
RIGHT_IDX = 3
LEFT_IDX = 6
# The following are accurate only after calling state.extract_features
# to mirror the state if necessary.
SWING_IDX = RIGHT_IDX
STANCE_IDX = LEFT_IDX

HIP_PITCH = 0
KNEE = 1
ANKLE = 2

def standardized_dofs(raw_dofs):
    return raw_dofs

def raw_dofs(standardized_dofs):
    return standardized_dofs

def virtual_torque_idx(standardized_idx):
    return standardized_idx + HIP_PITCH

def fix_Kd_idx(standardized_idx):
    return standardized_idx + KNEE

PELVIS_BODYNODE_IDX = 2
RIGHT_BODYNODE_IDX = 3
LEFT_BODYNODE_IDX = 6
THIGH_BODYNODE_OFFSET = 0
SHIN_BODYNODE_OFFSET = 1
FOOT_BODYNODE_OFFSET = 2

L_THIGH =    0.45
L_SHIN =   0.50
L_FOOT =   0.20
FOOT_RADIUS = 0.06

KP_GAIN = [200.0]*6
KD_GAIN = [15.0]*6

ALLOWED_COLLISION_IDS = [5,8] # The two feet

def hip_dofs_from_transform(_, transform):
    euler = libtransform.euler_from_matrix(transform, 'rxyz')
    return euler[2]

def root_dofs_from_transform(transform):
    euler = libtransform.euler_from_matrix(transform, 'rxyz')
    translation = transform[0:3,3]
    # The -1.25 is a hack (see ik.get_dofs())
    return np.array([translation[0], translation[1]-1.25, euler[2]])

DEFAULT_GROUND_WIDTH = 0.5
DOT_RADIUS = 0.08

def ankle_dofs_from_transform(_, relative_transform):
    euler = libtransform.euler_from_matrix(relative_transform, 'rzyx')
    return euler[0]

def foot_transform_from_angles(_, pitch, __):
    return libtransform.euler_matrix(pitch, 0, 0, 'rzxy')
