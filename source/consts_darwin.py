import numpy as np
import pydart2.utils.transformations as libtransform

from consts_common3D import *
Q_DIM_RAW = 26
SIMULATION_RATE = 0.002
GROUND_LEVEL = -0.35

skel_file = "skel/darwinmodel/darwin_ground.skel"
robot_model = "skel/darwinmodel/robotis_op2.urdf"

perm = [3,4,5,2,1,0,
        22,20,21,23,24,25,
        16,14,15,17,18,19,
        12] # Darwin actually doesn't have a torso roll actuator... TODO cleanup this interface

# These are applied after the permutation
sign_switches = [6,7,9,13,16]

def convert_euler(eulers, start_config, end_config):
    rotation = libtransform.euler_matrix(*eulers, start_config)
    return libtransform.euler_from_matrix(rotation, end_config)

# Maps the raw agent state to a standardized ordering of DOFs with standardized signs.
# May involve a nonlinear transformation to standardize the use of Euler angles.
# Some DOFs redundant to the FSM controller will be ignored, so this may return a shorter
# array than it receives. The inverse operation is `raw_dofs`.
def standardized_dofs(raw_dofs):
    r = raw_dofs[perm]
    r[sign_switches] *= -1
    r[3:6] = convert_euler(r[3:6], 'rxyz', 'ryzx')
    r[6:9] = convert_euler(r[6:9], 'rxyz', 'ryzx')
    r[12:15] = convert_euler(r[12:15], 'rxyz', 'ryzx')
    return r

def raw_dofs(standardized_dofs):
    r = standardized_dofs.copy()
    r[3:6] = convert_euler(r[3:6], 'ryzx', 'rxyz')
    r[6:9] = convert_euler(r[6:9], 'ryzx', 'rxyz')
    r[12:15] = convert_euler(r[12:15], 'ryzx', 'rxyz')
    r[sign_switches] *= -1
    base = np.zeros(Q_DIM_RAW)
    for i,j in enumerate(perm):
        base[j] = r[i]
    return base

def virtual_torque_idx(standardized_idx):
    if standardized_idx == RIGHT_IDX:
        return 22
    elif standardized_idx == LEFT_IDX:
        return 16
    else:
        raise "Invalid stance index"

def fix_Kd_idx(standardized_idx):
    if standardized_idx == RIGHT_IDX:
        return 23
    elif standardized_idx == LEFT_IDX:
        return 17
    else:
        raise "Invalid stance index"

CONTROL_BOUNDS = np.array([-0.01*np.ones(20,), 0.05*np.ones(20,)])
CONTROL_BOUNDS[0,[8,9,14,15]] = -0.01
CONTROL_BOUNDS[1,[8,9,14,15]] = 0.01
CONTROL_BOUNDS[0,[10,16]] = -0.20
CONTROL_BOUNDS[1,[10,16]] = 0.20
CONTROL_BOUNDS *= 4

KP_GAIN = np.array([154.019]*20)
KD_GAIN = np.array([0.1002]*20)

DEFAULT_ZOOM = 2.0

PELVIS_BODYNODE_IDX = 2
LEFT_BODYNODE_IDX = 3
RIGHT_BODYNODE_IDX = 6
THIGH_BODYNODE_OFFSET = 0
FOOT_BODYNODE_OFFSET = 2

L_FOOT =   0.21
FOOT_RADIUS = 0.049 # Not really a radius; it's a box....

ALLOWED_COLLISION_IDS = [
        RIGHT_BODYNODE_IDX + FOOT_BODYNODE_OFFSET,
        LEFT_BODYNODE_IDX + FOOT_BODYNODE_OFFSET] # The two feet

# The relative transform of the thigh when all DOFs of the joint are set to zero
LEFT_THIGH_RESTING_RELATIVE_TRANSFORM = np.array([[  3.26794897e-07,   0.00000000e+00,  -1.00000000e+00,
         -4.00323748e-08],
       [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
         -3.50000000e-02],
       [  1.00000000e+00,   0.00000000e+00,   3.26794897e-07,
         -1.22500000e-01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])

RIGHT_THIGH_RESTING_RELATIVE_TRANSFORM = np.array([[  3.26794897e-07,   0.00000000e+00,  -1.00000000e+00,
          4.00323748e-08],
       [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
         -3.50000000e-02],
       [  1.00000000e+00,   0.00000000e+00,   3.26794897e-07,
          1.22500000e-01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])
LEFT_RRT = np.linalg.inv(LEFT_THIGH_RESTING_RELATIVE_TRANSFORM)
RIGHT_RRT = np.linalg.inv(RIGHT_THIGH_RESTING_RELATIVE_TRANSFORM)

def hip_dofs_from_transform(transform):
    euler = libtransform.euler_from_matrix(transform, 'rzyx')
    hip_dofs = np.array([euler[2], euler[1], -euler[0]])
    return hip_dofs

def root_dofs_from_transform(transform):
    euler = libtransform.euler_from_matrix(transform, 'ryzx')
    translation = transform[0:3,3]
    dofs = np.zeros(6)
    dofs[3:6] = [euler[1], euler[0], euler[2]]
    dofs[0:3] = translation
    return dofs
