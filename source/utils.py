import numpy as np
from IPython import embed
import libtransform

def convert_euler(eulers, start_config, end_config):
    rotation = libtransform.euler_matrix(eulers[0], eulers[1], eulers[2], start_config)
    return libtransform.euler_from_matrix(rotation, end_config)

def heading_from_vector(d, branch=0.0):
        planar_d = d.copy()
        planar_d[1] = 0.0 # Project to X-Z plane
        planar_d = planar_d / np.linalg.norm(planar_d)
        heading = np.arccos(planar_d[0])
        # Remember Z sign is flipped (TODO maybe I should switch the sign of yaw?)
        if planar_d[2] > 0:
            heading *= -1
        # Adjust by multiples of 2*pi until |heading - branch| <= pi
        k, b = divmod(branch, 2*np.pi)
        if heading - b <= -np.pi:
            heading += (k+1)*2*np.pi
        else:
            heading += k*2*np.pi
        return planar_d, heading

def reward(controller, state):
    score = 1.0
    score -= np.linalg.norm(state.stance_heel_location() - state.stance_platform())
    # Balance penalty: Having the COM far away from the stance foot is bad
    com_dist = controller.distance_to_go(state.pose()[:3])
    heel_dist = controller.distance_to_go(state.stance_heel_location())
    d = com_dist - heel_dist
    v = controller.speed(state.dq())
    time_before_com_passes_stance_heel = d/v
    # TODO: is this right for the 2D model?
    ideal_time = 0.13
    diff = np.abs(time_before_com_passes_stance_heel - ideal_time)
    # If CoM is too far behind, we fall over backwards eventually; if too far forward,
    # we will fall forwards. Penalize this.
    score -= max(0, diff-0.10)
    if state.consts.BRICK_DOF == 6:
        slip_angle = np.abs(state.pose()[state.consts.ROOT_YAW])
        if slip_angle > np.pi/20:
            score -= (slip_angle-np.pi/20) # Hinge loss
    return score

def rotmatrix(theta):
    # Note we're rotating in the X-Z plane instead of X-Y, so some signs are weird.
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [            0, 1,              0],
                     [np.sin(theta), 0,  np.cos(theta)]])

def build_mask(length, true_indices):
    mask = np.zeros(length, dtype=np.bool_)
    mask[true_indices] = True
    return mask

if __name__ == "__main__":
    _, r1 = heading_from_vector([1,0,0], 0)
    print(np.allclose(r1, 0))
    _, r2 = heading_from_vector([10,0,0], 4*np.pi + 1)
    print(np.allclose(r2, 4*np.pi))
    _, r3 = heading_from_vector([10,-900,0], 4*np.pi - 1)
    print(np.allclose(r3, 4*np.pi))
    _, r4 = heading_from_vector([-1,0,-1], -np.pi)
    print(np.allclose(r4, -5/4*np.pi))
    _, r5 = heading_from_vector([0,0,1], -3*np.pi)
    print(np.allclose(r5, -2.5*np.pi))
