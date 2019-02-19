import numpy as np
import pydart2.utils.transformations as libtransform

from consts_common3D import *
Q_DIM_RAW = 21
DEFAULT_ZOOM = 5.0
GROUND_LEVEL = -0.9

skel_file = "skel/HumanSkel/kima_human_box_armless_visiblecollisionboxes.skel"
SIMULATION_RATE = 1.0 / 2000.0 # seconds

perm = [0,1,2,3,4,5, # Brick DOFs
        13,14,12,15,16,17, # Right leg
        7,8,6,9,10,11, # Left leg
        18] # Torso roll

sign_switches = [6,11,12]

GRAVITY_Y = True
def convert_root(q):
    return q
def inverse_convert_root(q):
    return q

def standardized_dofs(raw_dofs):
    r = np.array(raw_dofs)
    r[sign_switches] *= -1
    r = r[perm]
    return r

def raw_dofs(standardized_dofs):
    r = np.array(standardized_dofs)
    base = np.zeros(Q_DIM_RAW)
    for i,j in enumerate(perm):
        base[j] = r[i]
    base[sign_switches] *= -1
    return base

def virtual_torque_idx(standardized_idx):
    if standardized_idx == RIGHT_IDX:
        return 14
    elif standardized_idx == LEFT_IDX:
        return 8
    else:
        raise "Invalid stance index"

def fix_Kd_idx(standardized_idx):
    if standardized_idx == RIGHT_IDX:
        return 15
    elif standardized_idx == LEFT_IDX:
        return 9
    else:
        raise "Invalid stance index"

CONTROL_BOUNDS = 1000 * np.array([[-100]*Q_DIM, [100]*Q_DIM])

PELVIS_BODYNODE_IDX = 2
LEFT_BODYNODE_IDX = 3
RIGHT_BODYNODE_IDX = 6
THIGH_BODYNODE_OFFSET = 0
FOOT_BODYNODE_OFFSET = 2

# TODO this skel file has a lot of "transformation" values that might invalidate
# the current IK code.
L_THIGH =    0.42875
L_SHIN =   0.400875
L_FOOT =   0.21
FOOT_RADIUS = 0.049 # Not really a radius; it's a box....
DOT_RADIUS = 0.08

# Three hip, one knee, two ankle
leg_kp = [600, 600, 600, 200, 200, 100]
leg_kd = [70, 70, 70, 15, 15, 10]
# Two abdomen, one chest
trunk_kp = [600, 600, 300]
trunk_kd = [70, 70, 20]
KP_GAIN = leg_kp + leg_kp + trunk_kp
KD_GAIN = leg_kd + leg_kd + trunk_kd

ALLOWED_COLLISION_IDS = [
        RIGHT_BODYNODE_IDX + FOOT_BODYNODE_OFFSET,
        LEFT_BODYNODE_IDX + FOOT_BODYNODE_OFFSET] # The two feet

def hip_dofs_from_transform(transform):
    euler = libtransform.euler_from_matrix(transform, 'rzyx')
    hip_dofs = np.array([euler[1], euler[2], -euler[0]])
    return hip_dofs

def root_dofs_from_transform(transform):
    euler = libtransform.euler_from_matrix(transform, 'ryzx')
    translation = transform[0:3,3]
    dofs = np.zeros(6)
    dofs[3:6] = euler
    dofs[0:3] = translation
    return dofs
