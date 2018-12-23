import numpy as np
import pydart2.utils.transformations as libtransform

skel_file = "skel/HumanSkel/kima_human_box_armless_visiblecollisionboxes.skel"

perm = [0,1,2,4,3,5, # Brick DOFs
        14,13,12,15,16,17, # Right leg
        8,7,6,9,10,11, # Left leg
        18,19,20] # Everything else

# These are applied in standardized space
sign_switches = [8,14,17]

from consts_common3D import *

CONTROL_BOUNDS = 1000 * np.array([100]*Q_DIM)

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

# Three hip, one knee, two ankle
leg_kp = [600, 600, 600, 200, 200, 100]
leg_kd = [70, 70, 70, 15, 15, 10]
# Two abdomen, one chest
trunk_kp = [300, 300, 300]
trunk_kd = [20, 20, 20]
KP_GAIN = leg_kp + leg_kp + trunk_kp
KD_GAIN = leg_kd + leg_kd + trunk_kd

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
