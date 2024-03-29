from simple_3D_env import Simple3DEnv, test_pd_control, test_no_control, setup_dof_test
import numpy as np
import consts_darwin
from IPython import embed
from hardware.Sim2Robot import raw_darwin_init_state

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
            for limit in self.consts().LIMITS:
                if dof.name == limit[0]:
                    assert(dof.index == limit[1])
                    dof.set_position_lower_limit(limit[2])
                    dof.set_position_upper_limit(limit[3])

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

        skel.q = raw_darwin_init_state

        return skel

    def reset_world(self):
        super().reset_world()
        self.robot_skeleton.q = raw_darwin_init_state

if __name__ == "__main__":
    from pd_control import PDController
    env = DarwinEnv(PDController)
    setup_dof_test(env)
    test_pd_control(env, secs=2)
