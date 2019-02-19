import numpy as np
import pydart2.utils.transformations as libtransform
from utils import convert_euler

from consts_common3D import *
Q_DIM_RAW = 26
SIMULATION_RATE = 0.002
GROUND_LEVEL = -0.35
DEFAULT_ZOOM = 2.0

GRAVITY_Y = False

def convert_root(brick_dof):
    # Switch from Y axis gravity to Z axis gravity
    brick_dof[0:3] = np.array(brick_dof)[[0,2,1]]
    brick_dof[1] *= -1
    if len(brick_dof) == 6:
        brick_dof[3] -= np.pi/2
    return brick_dof

def inverse_convert_root(brick_dof):
    if len(brick_dof) == 6:
        brick_dof[3] += np.pi/2
    brick_dof[1] *= -1
    brick_dof[0:3] = np.array(brick_dof)[[0,2,1]]
    return brick_dof

skel_file = "skel/darwinmodel/darwin_ground.skel"
robot_model = "skel/darwinmodel/robotis_op2.urdf"

perm = [0,2,1,3,4,5,
        20,21,22,23,24,25,
        14,15,16,17,18,19,
        12] # Darwin actually doesn't have a torso roll actuator... TODO cleanup this interface

sign_switches = [1,4,14,15,18,20,21,22,23]

STANDARD_EULER_ORDER = 'ryzx'
RAW_EULER_ORDER_HIP = 'ryxz'
# Maps the raw agent state to a standardized ordering of DOFs with standardized signs.
# May involve a nonlinear transformation to standardize the use of Euler angles.
# Some DOFs redundant to the FSM controller will be ignored, so this may return a shorter
# array than it receives. The "inverse" operation is `raw_dofs`.
def standardized_dofs(raw_dofs):
    r = np.array(raw_dofs)
    r[sign_switches] *= -1
    r = r[perm]
    r[6:9] = convert_euler(r[6:9], RAW_EULER_ORDER_HIP, STANDARD_EULER_ORDER)
    r[12:15] = convert_euler(r[12:15], RAW_EULER_ORDER_HIP, STANDARD_EULER_ORDER)
    return r

def raw_dofs(standardized_dofs):
    r = np.array(standardized_dofs)
    r[6:9] = convert_euler(r[6:9], STANDARD_EULER_ORDER, RAW_EULER_ORDER_HIP)
    r[12:15] = convert_euler(r[12:15], STANDARD_EULER_ORDER, RAW_EULER_ORDER_HIP)
    base = np.zeros(Q_DIM_RAW)
    for i,j in enumerate(perm):
        base[j] = r[i]
    base[sign_switches] *= -1
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

CONTROL_BOUNDS = np.array([-7.5*np.ones(20,), 7.5*np.ones(20,)])

KP_GAIN = np.array([154.019]*20)
KD_GAIN = np.array([0.1002]*20)

FOOT_RADIUS = 0.004
L_FOOT = 0.104
RIGHT_BODYNODE_IDX = 22
LEFT_BODYNODE_IDX = 16
FOOT_BODYNODE_OFFSET = 5
THIGH_BODYNODE_OFFSET = 2
PELVIS_BODYNODE_IDX = 2
DOT_RADIUS = 0.01

def hip_dofs_from_transform(stance_idx, relative_transform):
    # For some reason, the relative transform is given in a coordinate system
    # where the X axis points down and the Y axis points forward.
    # (At least, true for the left leg. Maybe the right leg is different
    # or is using a left-handed coordinate system; I didn't check.)
    # Hence, we conjugate it to the coordinate system with X pointing forward
    # and Y pointing left.
    axis_correction = libtransform.euler_matrix(0.0, np.pi/2, np.pi/2, 'rzyx')
    corrected_transform = axis_correction.dot(relative_transform).dot(np.linalg.inv(axis_correction))
    euler = libtransform.euler_from_matrix(corrected_transform, 'rzyx')
    # I honestly do not understand why all of these signs are backwards,
    # or why the sign changes are different for each leg. I give up.
    if stance_idx == LEFT_IDX:
        return euler*np.array([-1,-1,-1])
    elif stance_idx == RIGHT_IDX:
        return euler*np.array([ 1, 1,-1])

def root_dofs_from_transform(transform):
    euler = libtransform.euler_from_matrix(transform, 'rzyx')
    translation = transform[0:3,3]
    dofs = np.zeros(6)
    dofs[3:6] = euler
    dofs[4] *= -1
    dofs[0:3] = inverse_convert_root(translation)
    return dofs
