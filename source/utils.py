import numpy as np
from IPython import embed

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
