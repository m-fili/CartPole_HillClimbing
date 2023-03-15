import numpy as np


class HillClimbing:
    """
    Maximization Problem.
    """

    def __init__(self, method='vanilla', noise_scale=0.1, n_candidates=1, up_rate=2, down_rate=0.5, max_noise=2,
                 min_noise=0.001):

        assert method in ['vanilla', 'steepest_ascent', 'simulated_annealing', 'adaptive_noise_scaling']

        self.x_best = None
        self.f_best = -np.inf
        self.noise_scale = noise_scale
        self.n_candidates = n_candidates
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.method = method

    def step(self, xs, fs):

        xs_new = None

        if self.method == 'vanilla':
            if fs[0] > self.f_best:
                self.x_best = xs[0]
                self.f_best = fs[0]
            xs_new = [self.x_best + np.random.normal(loc=0, scale=self.noise_scale, size=xs[0].shape)]

        if self.method == 'steepest_ascent':
            best_indx = np.argmax(fs)

            if fs[best_indx] > self.f_best:
                self.x_best = xs[best_indx]
                self.f_best = fs[best_indx]
            xs_new = [self.x_best + np.random.normal(0, self.noise_scale, size=xs[0].shape) for _ in
                      range(self.n_candidates)]

        if self.method == 'simulated_annealing':
            best_indx = np.argmax(fs)

            if fs[best_indx] > self.f_best:
                self.x_best = xs[best_indx]
                self.f_best = fs[best_indx]
                self.noise_scale /= self.down_rate
            xs_new = [self.x_best + np.random.normal(0, self.noise_scale, size=xs[0].shape) for _ in
                      range(self.n_candidates)]

        if self.method == 'adaptive_noise_scaling':
            best_indx = np.argmax(fs)

            if fs[best_indx] > self.f_best:
                self.x_best = xs[best_indx]
                self.f_best = fs[best_indx]
                self.noise_scale = max(self.noise_scale * self.down_rate, self.min_noise)
            else:
                self.noise_scale = min(self.noise_scale * self.up_rate, self.max_noise)

            xs_new = [self.x_best + np.random.normal(0, self.noise_scale, size=xs[0].shape) for _ in
                      range(self.n_candidates)]

        return xs_new
