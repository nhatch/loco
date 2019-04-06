from simple_3D_env import Simple3DEnv, test_pd_control, test_no_control, setup_dof_test
import numpy as np
import consts_darwin
from IPython import embed

class DarwinEnv(Simple3DEnv):
    def consts(self):
        return consts_darwin

    def load_model(self, world, model):
        skel = world.add_skeleton(model)
        skel.set_root_joint_to_trans_and_euler()
        skel.joints[1].set_axis_order('ZYX')
        return skel

    def load_robot(self, world):
        skel = self.load_model(world, self.consts().robot_model)
        doppelganger = self.load_model(world, self.consts().doppelganger_model)

        for dof in skel.dofs[6:]:
            dof.set_damping_coefficient(0.2165)
            # Copied limits from
            # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.822.6324&rep=rep1&type=pdf
            if dof.name == 'j_tibia_r':
                dof.set_position_lower_limit(0.)
                dof.set_position_upper_limit(130/180*np.pi)
            if dof.name == 'j_tibia_l':
                dof.set_position_upper_limit(0.)
                dof.set_position_lower_limit(-130/180*np.pi)
            if dof.name in ['j_ankle1_r', 'j_ankle1_l']:
                dof.set_position_lower_limit(-np.pi/3)
                dof.set_position_upper_limit(np.pi/3)
            if dof.name in ['j_ankle2_r', 'j_ankle2_l']:
                # Note this does not match the document above. I decreased the upper limit
                # in order to get bilateral symmetry.
                dof.set_position_lower_limit(-np.pi/6)
                dof.set_position_upper_limit(np.pi/6)
            if dof.name == 'j_thigh1_l':
                dof.set_position_lower_limit(-np.pi/3)
                dof.set_position_upper_limit(0.)
            if dof.name == 'j_thigh1_r':
                dof.set_position_lower_limit(0.)
                dof.set_position_upper_limit(np.pi/3)
            if dof.name == 'j_thigh2_l':
                dof.set_position_lower_limit(-100/180*np.pi)
                dof.set_position_upper_limit(np.pi/6)
            if dof.name == 'j_thigh2_r':
                dof.set_position_lower_limit(-np.pi/6)
                dof.set_position_upper_limit(100/180*np.pi)
            # I don't think we need to set the limits for hip yaw, since
            # it's unlikely that we'll hit the numbers pu blished in the above document.

        for j in skel.joints:
            j.set_position_limit_enforced(True)

        for body in skel.bodynodes:
            if body.name == "base_link":
                body.set_mass(0.001)
            if body.name == "MP_PMDCAMBOARD":
                body.set_mass(0.001)
            if body.name == "MP_BODY":
                body.set_mass(0.03)
            if body.name == "MP_BACK_L":
                body.set_mass(0.6)
            if body.name == "MP_BACK_R":
                body.set_mass(0.6)

        world.skeletons[0].bodynodes[0].set_friction_coeff(0.716)
        import curriculum as cur
        cur.SETTINGS_3D_EASY['z_mean'] /= 2

        return skel

if __name__ == "__main__":
    from pd_control import PDController
    env = DarwinEnv(PDController)
    setup_dof_test(env)
    test_pd_control(env, secs=2)
