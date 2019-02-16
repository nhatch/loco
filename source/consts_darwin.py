import numpy as np
import pydart2.utils.transformations as libtransform
from utils import convert_euler

from consts_common3D import *
Q_DIM_RAW = 26
SIMULATION_RATE = 0.002
GROUND_LEVEL = -0.35

skel_file = "skel/darwinmodel/darwin_ground.skel"
robot_model = "skel/darwinmodel/robotis_op2.urdf"

perm = [3,4,5,2,1,0,
        20,21,22,23,24,25,
        14,15,16,17,18,19,
        12] # Darwin actually doesn't have a torso roll actuator... TODO cleanup this interface

sign_switches = [14,15,18,20,21,22,23]

STANDARD_EULER_ORDER = 'ryzx'
RAW_EULER_ORDER_HIP = 'ryxz'
RAW_EULER_ORDER_ROOT = 'rxyz'
# Maps the raw agent state to a standardized ordering of DOFs with standardized signs.
# May involve a nonlinear transformation to standardize the use of Euler angles.
# Some DOFs redundant to the FSM controller will be ignored, so this may return a shorter
# array than it receives. The inverse operation is `raw_dofs`.
def standardized_dofs(raw_dofs):
    r = np.array(raw_dofs)
    r[sign_switches] *= -1
    r = r[perm]
    r[3:6] = convert_euler(r[3:6], RAW_EULER_ORDER_ROOT, STANDARD_EULER_ORDER)
    r[6:9] = convert_euler(r[6:9], RAW_EULER_ORDER_HIP, STANDARD_EULER_ORDER)
    r[12:15] = convert_euler(r[12:15], RAW_EULER_ORDER_HIP, STANDARD_EULER_ORDER)
    return r

def raw_dofs(standardized_dofs):
    r = np.array(standardized_dofs)
    r[3:6] = convert_euler(r[3:6], STANDARD_EULER_ORDER, RAW_EULER_ORDER_ROOT)
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

CONTROL_BOUNDS = np.array([-0.01*np.ones(20,), 0.05*np.ones(20,)])
CONTROL_BOUNDS[0,[8,9,14,15]] = -0.01
CONTROL_BOUNDS[1,[8,9,14,15]] = 0.01
CONTROL_BOUNDS[0,[10,16]] = -0.20
CONTROL_BOUNDS[1,[10,16]] = 0.20
CONTROL_BOUNDS = np.array([-7.5*np.ones(20,), 7.5*np.ones(20,)])

KP_GAIN = np.array([154.019]*20)
KD_GAIN = np.array([0.1002]*20)

DEFAULT_ZOOM = 2.0
