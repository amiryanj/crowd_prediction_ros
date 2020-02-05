import os
import numpy as np
from math import sqrt
from sklearn.metrics.pairwise import euclidean_distances
# from followbot.temp.parser_eth import ParserETH
from bisect import bisect_left, bisect_right


class PcfPattern:
    def __init__(self, sigma_=0.25, compat_thresh=0.1):
        self.pcf_values = []
        self.rad_values = []
        self.get_pcf = []
        self.sigma = sigma_
        self.compat_threshold = compat_thresh

    def update(self, points, rng=np.arange(0.2, 10, 0.05)):
        N = len(points)

        self.rad_values = rng
        self.pcf_values = np.empty(len(rng), dtype=np.float)
        self.get_pcf = lambda r: self.pcf_values[bisect_right(self.rad_values, r)]

        dists = euclidean_distances(points)
        dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N-1)
        # return dists

        for ii, rr in enumerate(rng):
            A_r = np.pi * (rr + 0.2) ** 2 - np.pi * max((rr - 0.2), 0) ** 2

            dists_sqr = np.power(dists_wo_diag - rr, 2)
            dists_exp = np.exp(-dists_sqr / (self.sigma ** 2)) / (sqrt(np.pi) * self.sigma)
            pcf_r = np.sum(dists_exp) / N**2 / 2. /A_r # because each dist is repeated twice

            self.pcf_values[ii] = pcf_r

    def compatible(self, points):
        if len(self.pcf_values) == 0:
            print('Error! First call update pcf')
            return False

        N = len(points)
        dists = euclidean_distances(points)
        dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N - 1)

        for ii, rr in enumerate(self.rad_values):
            A_r = np.pi * (rr + 0.2) ** 2 - np.pi * max((rr - 0.2), 0) ** 2

            dists_sqr = np.power(dists_wo_diag - rr, 2)
            dists_exp = np.exp(-dists_sqr / (self.sigma ** 2)) / (sqrt(np.pi) * self.sigma)
            pcf_r = np.sum(dists_exp) / N ** 2 / 2. / A_r  # because each dist is repeated twice

            if pcf_r > self.pcf_values[ii] + self.compat_threshold:
                return False
        return True


# ===========================
# === Synthesis Algorithm ===
# ===========================
class DartThrowing:
    def __init__(self):
        self.pdf_accum = []
        self.map_to_world_coord = lambda : 0

    def set_pdf_grid(self, pdf_grid, map_to_world_coord):
        sum = pdf_grid.reshape(-1).sum()
        self.pdf_accum = pdf_grid.reshape(-1).cumsum() / (sum + 1E-19)
        self.map_to_world_coord = map_to_world_coord

    def random_sample(self):
        rnd = np.random.rand()
        index = bisect_left(self.pdf_accum, rnd)
        x, y = self.map_to_world_coord(index)
        return (x, y)

    # TODO:
    #  test functions
    def dart_with_social_space(self, points, thresh=0.5):
        for kk in range(100):  # number of tries
            new_point = np.random.rand(2) * np.array([3.5, 7]) + np.array([0.25, 0.5])
            if len(points) == 0:
                return new_point
            new_point_repeated = np.repeat(new_point.reshape((1, 2)), len(points), axis=0)
            dists = np.linalg.norm(np.array(points) - new_point_repeated, axis=1)
            if min(dists) > thresh:
                return new_point

        # FIXME: otherwise ignore the distance and return sth
        return np.random.uniform(2) * np.array([3.5, 7]) + np.array([0.25, 0.5])

    def dart_throwing(self, n, target_pcf, prior_pom):
        error_eps = 0.5
        points = []
        counter = 0
        while len(points) < n:
            new_pt = sample_pom(pdf_grid=prior_pom)
            points_tmp = [points[:], new_pt]
            new_pcf_lut = pcf(points_tmp)
            new_pcf_vals = np.array(sorted(new_pcf_lut.items()))[:, 1]
            err = np.clip(new_pcf_vals - target_pcf, 0, 1000)  # skip values smaller than target values
            max_err = max(err)
            if max_err < error_eps:
                points.append(new_pt)

        return points


def get_rand_n_agents(hist):
    hist_cum = np.cumsum(hist) / np.sum(hist)
    rnd = np.random.random(1)
    for ii, val in enumerate(hist_cum):
        if rnd < val:
            return ii


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    # crowd_by_click()
    pcf_pattern = PcfPattern()


