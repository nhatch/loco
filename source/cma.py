import numpy as np
from scipy.special import gamma
from IPython import embed

class CMA:
    def __init__(self, f, initial_mean, initial_sigma, initial_cov,
            extra_lambda=0, alpha_cov=2.):
        self.f = f
        self.N = initial_mean.shape[0]
        self.set_consts(extra_lambda, alpha_cov)
        self.mean = initial_mean.reshape((-1,1))
        self.sigma = initial_sigma
        self.cov = initial_cov
        self.cov_path = np.zeros((self.N,1))
        self.std_path = np.zeros((self.N,1))
        self.generation = 0

    def set_consts(self, extra_lambda, alpha_cov):
        self.LAMBDA = int(4 + np.floor(3 * np.log(self.N))) + extra_lambda
        RECOMB_W = np.array([np.log(self.LAMBDA+1) - np.log(2*(i+1))
            for i in range(self.LAMBDA)])
        self.MU = int(np.floor(self.LAMBDA / 2))
        RECOMB_W[:self.MU] /= RECOMB_W[:self.MU].sum()
        RECOMB_W[self.MU:] /= -RECOMB_W[self.MU:].sum()
        # Should be approx lambda/4
        self.MU_EFF =  1/(RECOMB_W[:self.MU]**2).sum()
        MU_EFF_MINUS = 1/(RECOMB_W[self.MU:]**2).sum()

        self.C_MEAN = 1.0

        # Expected norm of a random normal vector with zero mean and identity covariance
        # Should be approx sqrt(N)
        self.E_NORM = np.sqrt(2) * gamma((self.N+1)/2) / gamma(self.N/2)
        # Should be between 1/N and 1/sqrt(N)
        self.C_SIGMA = (self.MU_EFF+2) / (self.N + self.MU_EFF + 5)
        # Should be approx 1
        self.D_SIGMA = 1 + self.C_SIGMA
        potential_addition = np.sqrt((self.MU_EFF - 1)/(self.N + 1)) - 1
        if potential_addition > 0:
            self.D_SIGMA += potential_addition

        # Should be between 1/N and 1/sqrt(N)
        self.C_C = (4+self.MU_EFF/self.N) / (self.N+4+2*self.MU_EFF/self.N)
        # Should be approx 2/N^2
        self.C_1 = alpha_cov/((self.N+1.3)**2 + self.MU_EFF)
        thingy = (self.MU_EFF-2+1/self.MU_EFF)/((self.N+2)**2 + alpha_cov*self.MU_EFF/2)
        # Should be approx MU_EFF/N^2 (and less than 1)
        self.C_MU = min(1-self.C_1, alpha_cov * thingy)

        rescale_neg_w = np.min([
                1 + self.C_1/self.C_MU,
                1 + 2*MU_EFF_MINUS / (self.MU_EFF + 2),
                (1 - self.C_1 - self.C_MU) / (self.N * self.C_MU)
            ])
        RECOMB_W[self.MU:] *= rescale_neg_w
        self.RECOMB_W = RECOMB_W.reshape((-1,1))

        # For calculating h_sigma
        self.RHS = (1.4 + 2/(self.N+1))*self.E_NORM

    def iter(self):
        self.recalc_eig()
        y = self.sample_population()
        y_w = y.dot(self.RECOMB_W*(self.RECOMB_W > 0))
        new_mean = self.update_mean(y_w)
        new_std_path = self.update_std_path(y_w)
        h_sigma = self.calc_h_sigma(new_std_path)
        new_cov_path = self.update_cov_path(y_w, h_sigma)
        new_cov = self.update_cov(y, new_cov_path, h_sigma)
        new_sigma = self.update_sigma(new_std_path)

        self.mean = new_mean
        self.cov_path = new_cov_path
        self.std_path = new_std_path
        self.cov = new_cov
        self.sigma = new_sigma
        self.generation += 1

    def calc_h_sigma(self, new_std_path):
        LHS = np.linalg.norm(new_std_path) / np.sqrt(1-(1-self.C_SIGMA)**(2*self.generation+2))
        return LHS < self.RHS

    def recalc_eig(self):
        # D**2 and B in the notation of https://arxiv.org/pdf/1604.00772.pdf
        D2, B = np.linalg.eig(self.cov)
        # In case for numerical reasons there are imaginary values
        self.B = B.astype(np.double)
        self.D = (D2.astype(np.double)**0.5).reshape((self.N, 1))

    def sample_population(self):
        z = np.random.randn(self.LAMBDA, self.N).T # We want column vectors
        y = self.B.dot(self.D * z)
        pop = self.sigma*y + self.mean
        # Sort from lowest to highest
        f_vals = [self.f(x, render=None) for x in pop.T]
        perm = np.argsort(f_vals)
        return (y.T)[perm].T

    def update_mean(self, y_w):
        update = self.C_MEAN * self.sigma * y_w
        return update + self.mean

    def update_cov_path(self, y_w, h_sigma):
        C_C = self.C_C
        new_cov_path = (1-C_C)*self.cov_path
        if h_sigma:
            new_cov_path += np.sqrt(C_C*(2-C_C)*self.MU_EFF) * y_w
        return new_cov_path

    def update_cov(self, y, new_cov_path, h_sigma):
        C_sqrtinv = self.B.dot(self.B.T / self.D)
        rescale_neg_w = self.N / (np.dot(C_sqrtinv, y)**2).sum()
        W_CIRC = self.RECOMB_W.copy()
        W_CIRC[self.MU:] *= rescale_neg_w
        rank_mu_update = self.C_MU * np.dot(y*W_CIRC.T, y.T)
        rank_one_update = self.C_1 * new_cov_path.dot(new_cov_path.T)
        cov_decay = 1 - self.C_1 - self.C_MU*self.RECOMB_W.sum()
        if not h_sigma:
            cov_decay += self.C_1*self.C_C*(2-self.C_C)
        return cov_decay*self.cov + rank_one_update + rank_mu_update

    def update_std_path(self, y_w):
        C_sqrtinv = self.B.dot(self.B.T / self.D)
        d = C_sqrtinv.dot(y_w)
        C_S = self.C_SIGMA
        return (1-C_S)*self.std_path + np.sqrt(C_S*(2-C_S)*self.MU_EFF) * d

    def update_sigma(self, new_std_path):
        ratio = np.linalg.norm(new_std_path) / self.E_NORM
        return self.sigma * np.exp(self.C_SIGMA / self.D_SIGMA * (ratio - 1))



if __name__ == "__main__":
    def f(x, render=None):
        return np.linalg.norm(x)**2
    initial_mean = np.ones(20)
    initial_sigma = 0.5
    initial_cov = np.eye(20)
    opzer = CMA(f, initial_mean, initial_sigma, initial_cov)
    for i in range(50):
        print(np.linalg.norm(opzer.mean))
        opzer.iter()

