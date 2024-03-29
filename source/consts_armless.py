import numpy as np
import libtransform

from consts_common3D import *
Q_DIM_RAW = 21
DEFAULT_ZOOM = 5.0
GROUND_LEVEL = -0.9

skel_file = "skel/HumanSkel/kima_human_box_armless_visiblecollisionboxes.skel"
SIMULATION_FREQUENCY = 2000
CONTROL_FREQUENCY = 50
REAL_TIME_STEPS_PER_RENDER = 25
OBSERVE_TARGET = True
LIFTOFF_DURATION = 0.3

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

def clip(raw_dofs):
    return raw_dofs

def virtual_torque_idx(standardized_idx):
    if standardized_idx == RIGHT_IDX:
        return 14
    elif standardized_idx == LEFT_IDX:
        return 8
    else:
        raise "Invalid stance index"

CONTROL_BOUNDS = 1000 * np.array([[-100]*Q_DIM, [100]*Q_DIM])

PELVIS_BODYNODE_IDX = 2
LEFT_BODYNODE_IDX = 3
RIGHT_BODYNODE_IDX = 6
THIGH_BODYNODE_OFFSET = 0
SHIN_BODYNODE_OFFSET = 1
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

def hip_dofs_from_transform(_, transform):
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

def ankle_dofs_from_transform(_, relative_transform):
    euler = libtransform.euler_from_matrix(relative_transform, 'rxzy')
    dofs = np.array([euler[0], -euler[1]])
    return dofs

def foot_transform_from_angles(_, pitch, roll):
    correction = libtransform.euler_matrix(0, -np.pi/2, 0, 'rxyz')
    return correction.dot(libtransform.euler_matrix(pitch, 0, roll, 'rzxy'))

BASE_GAIT = np.array([0.14, 0.5, 0.2, -0.1, 0.2,
                      0.4, -1.1,   0.0, -0.05,
                      -0.0, -0.00, 0.1, -0.1,
                      0.5, 0.2, 0.0, 0.0, 0.0, 0.0,
                      0.0, # This one is only used for Darwin
                      0.0, 0.0])
