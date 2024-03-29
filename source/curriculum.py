import numpy as np
from IPython import embed
import simbicon_params as sp
import consts_2D as c2d
import consts_common3D as c3d
from utils import build_mask

# This defines a mask for the environment features which will be used to limit
# the dimensionality of training.
oq_2d = build_mask(c2d.Q_DIM, [c2d.X, c2d.Y, c2d.SWING_IDX+c2d.HIP_PITCH, c2d.STANCE_IDX+c2d.HIP_PITCH, c2d.STANCE_IDX+c2d.KNEE])
ot_2d = build_mask(3, [c2d.X, c2d.Y])
# Robot DOF state and velocity; stance heel location, previous target, target before that,
# then current target.
o2d = np.concatenate([oq_2d, oq_2d,
                      ot_2d, ot_2d, ot_2d, ot_2d])

TRAIN_SETTINGS_2D = {
    'n_trajectories': 4,
    'n_dirs': 4,
    'tol': 0.02,
    'max_iters': 5,
    'controllable_params':
        build_mask(sp.N_PARAMS,
            [
            sp.IK_GAIN,
            #sp.STANCE_ANKLE_RELATIVE,
            sp.UP_IDX+sp.SWING_ANKLE_RELATIVE,
            sp.UP_IDX+sp.SWING_HIP_WORLD,
            sp.UP_IDX+sp.SWING_KNEE_RELATIVE,
            #sp.UP_IDX+sp.STANCE_KNEE_RELATIVE,
            ]),
    'observable_features': o2d,
    'model_class': 'linear',
    }

cp_orig = TRAIN_SETTINGS_2D['controllable_params']
col = cp_orig.reshape((-1,1))
row = np.concatenate((o2d, [True])).reshape((1,-1))
cp_baseline_noexpert = np.dot(col, row) # Outer product

TRAIN_SETTINGS_BASELINE_NOEXPERT = {
    'n_dirs': 8,
    'tol': 0.00, # Not actually used
    'controllable_params': cp_baseline_noexpert,
    'rs_eps': 0.1,
    'rs_step_size': 0.01,
    }

o2d_plus = np.ones_like(o2d) # All features are observable

TRAIN_SETTINGS_2D_PLUS = {**TRAIN_SETTINGS_2D,
    'model_class': 'quadratic',
    }

TRAIN_SETTINGS_2D_NOCUR_FIRST = {**TRAIN_SETTINGS_2D_PLUS,
    'n_trajectories': 32,
    }

TRAIN_SETTINGS_2D_NOCUR_NEXT = {**TRAIN_SETTINGS_2D_PLUS,
    'n_trajectories': 16,
    }

SETTINGS_2D_EASY = {
    'use_stepping_stones': True,
    'ground_length': 0.1,
    'ground_width': 0.5,
    'dist_mean': 0.45,
    'dist_spread': 0.15,
    'n_steps': 16,
    'termination_tol': 0.07,
    'z_mean': 0.0,
    'z_spread': 0.0,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

SETTINGS_2D_HARD = {**SETTINGS_2D_EASY,
    'dist_spread': 0.3,
    'y_mean': 0.05,
    'y_spread': 0.1,
    }

oq_3d = build_mask(c3d.Q_DIM,
        [c3d.X, c3d.Z, c3d.ROOT_PITCH, c3d.ROOT_YAW, c3d.ROOT_ROLL,
        c3d.SWING_IDX+c3d.HIP_PITCH, c3d.SWING_IDX+c3d.HIP_ROLL,
        c3d.STANCE_IDX+c3d.HIP_PITCH, c3d.STANCE_IDX+c3d.HIP_ROLL,
        ])
ot_3d = build_mask(3, [c3d.X, c3d.Z])
o3d = np.concatenate([oq_3d, oq_3d,
                      ot_3d, ot_3d, ot_3d, ot_3d])

TRAIN_SETTINGS_3D = {
    'n_trajectories': 4,
    'n_dirs': 8,
    'tol': 0.05,
    'max_iters': 8,
    'controllable_params':
        build_mask(sp.N_PARAMS,
            [
            sp.IK_GAIN,
            sp.UP_IDX+sp.SWING_ANKLE_RELATIVE,
            #sp.STANCE_HIP_ROLL_EXTRA,
            #sp.POSITION_BALANCE_GAIN_LAT,
            sp.UP_IDX+sp.SWING_HIP_WORLD,
            #sp.STANCE_YAW,
            sp.UP_IDX+sp.SWING_KNEE_RELATIVE,
            ]),
    'observable_features': o3d,
    'model_class': 'linear',
    }

TRAIN_SETTINGS_3D_PLUS = {**TRAIN_SETTINGS_3D,
    'model_class': 'quadratic',
    }

SETTINGS_3D_FIRST = {
    'use_stepping_stones': True,
    'ground_length': 0.3,
    'ground_width': 0.2,
    'dist_mean': 0.35,
    'dist_spread': 0.0,
    'termination_tol': 0.12,
    'n_steps': 1,
    'z_mean': 0.4,
    'z_spread': 0.0,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

SETTINGS_3D_EASY = {**SETTINGS_3D_FIRST,
    'n_steps': 16,
    }

SETTINGS_3D_HARD = {**SETTINGS_3D_EASY,
    'dist_spread': 0.2,
    'z_spread': 0.1,
    }

