
TRAIN_SETTINGS_2D = {
    'n_trajectories': 3,
    'n_dirs': 4,
    'tol': 0.02,
    'max_intolerable_steps': 3,
    }

SETTINGS_2D_EASY = {
    'use_stepping_stones': False,
    'dist_mean': 0.47,
    'dist_spread': 0.3,
    'n_steps': 16,
    'tol': 0.02,
    'z_mean': 0.0,
    'z_spread': 0.0,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

SETTINGS_2D_HARD = {**SETTINGS_2D_EASY,
    'use_stepping_stones': True,
    'ground_length': 0.1,
    'ground_width': 0.5,
    'y_mean': 0.05,
    'y_spread': 0.1,
    }

TRAIN_SETTINGS_3D = {
    'n_trajectories': 3,
    'n_dirs': 8,
    'tol': 0.05,
    'max_intolerable_steps': 3,
    }

TRAIN_SETTINGS_3D_PRECISE = {**TRAIN_SETTINGS_3D,
    'tol': 0.02,
    }

SETTINGS_3D_EASY = {
    'use_stepping_stones': False,
    'dist_mean': 0.35,
    'dist_spread': 0.0,
    'early_termination_tol': 0.05,
    'n_steps': 16,
    'z_mean': 0.4,
    'z_spread': 0.0,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

SETTINGS_3D_MEDIUM = {**SETTINGS_3D_EASY,
    'dist_spread': 0.2,
    'z_spread': 0.1,
    }

SETTINGS_3D_HARD = {**SETTINGS_3D_MEDIUM,
    'dist_spread': 0.5,
    'z_spread': 0.2,
    }

SETTINGS_3D_HARDER = {**SETTINGS_3D_HARD,
    'use_stepping_stones': True,
    'ground_length': 0.3,
    'ground_width': 0.2,
    }

