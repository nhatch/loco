from simple_3D_env import Simple3DEnv, test_pd_control, test_no_control, setup_dof_test
import consts_darwin
from IPython import embed

class DarwinEnv(Simple3DEnv):
    def consts(self):
        return consts_darwin

    def load_robot(self, world):
        skel = world.add_skeleton(self.consts().robot_model)
        for dof in skel.dofs[6:]:
            dof.set_damping_coefficient(0.2165)
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

        return skel

if __name__ == "__main__":
    from pd_control import PDController
    env = DarwinEnv(PDController)
    #setup_dof_test(env)
    test_pd_control(env, secs=5)
