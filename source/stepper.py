import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from IPython import embed

BRICK_DOF = 3 # We're in 2D
#KP_GAIN = 200.0
#KD_GAIN = 15.0
KP_GAIN = 200.0
KD_GAIN = 20.0
TIME_STEP = 0.002 # seconds
FRAME_SKIP = 15


class Controller:
    def __init__(self, skel, control_bounds):
        self.skel = skel
        self.target_q = np.zeros(self.skel.ndofs)
        self.inactive = False
        self.Kp = np.array([0.0] * BRICK_DOF + [KP_GAIN] * (self.skel.ndofs - BRICK_DOF))
        self.Kd = np.array([0.0] * BRICK_DOF + [KD_GAIN] * (self.skel.ndofs - BRICK_DOF))

        # Keep the same control bounds as in the original DART environment, for fair comparison
        self.control_bounds = control_bounds
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])

    def compute(self):
        if self.inactive:
            return np.zeros_like(self.Kp)
        control = -self.Kp * (self.skel.q - self.target_q) - self.Kd * self.skel.dq
        clamped_control = control[3:]
        # TODO why is the control here not achieving the same footstep placement we saw before?
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.skel.ndofs)
        tau[3:] = clamped_control * self.action_scale
        return tau

class DartStepperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        obs_dim = 17

        dart_env.DartEnv.__init__(self, 'walker2d.skel', 4, obs_dim, self.control_bounds, disableViewer=False)

        utils.EzPickle.__init__(self)

        self.dots = [self.dart_world.add_skeleton('./dot.sdf') for _ in range(2)]
        self.robot_skeleton.set_controller(Controller(self.robot_skeleton, self.control_bounds))

    def _step(self, a):
        pre_state = [self.state_vector()]

        self.robot_skeleton.controller.target_q[BRICK_DOF:] = a

        posbefore = self.robot_skeleton.q[0]
        tau = np.zeros(self.robot_skeleton.ndofs) # Let the Controller handle this
        self.do_simulation(tau, FRAME_SKIP)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        reward = vel
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty'''

        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #            (height > .8) and (height < 2.0) and (abs(ang) < 1.0))
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all())

        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.put_dot(0.5, 0, 0)
        self.put_dot(1.0, 0, 1)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

    def put_dot(self, x, y, idx=0):
        dot = self.dots[idx]
        q = dot.q
        q[3] = x
        q[4] = y
        dot.q = q

from gym.envs.registration import register
register(
        id='Stepper-v0',
        entry_point='stepper:DartStepperEnv',
        max_episode_steps=150,
        reward_threshold=1.99
        )
