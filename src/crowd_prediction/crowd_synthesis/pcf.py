import os
import numpy as np
from math import sqrt
from sklearn.metrics.pairwise import euclidean_distances
# from followbot.temp.parser_eth import ParserETH
from bisect import bisect_left, bisect_right


class PcfPattern:
    def __init__(self, sigma_=0.25, compat_thresh=0.02):
        self.points = []
        self.pcf_values = []
        self.rad_values = []
        self.get_pcf = []
        self.sigma = sigma_
        self.compat_threshold = compat_thresh
        self.area_r = lambda r : np.pi * (r + 0.2) ** 2 - np.pi * max((r - 0.2), 0) ** 2

    def pcf_r(self, dist_matrix, rr):
        dists_sqr = np.power(dist_matrix - rr, 2)
        dists_exp = np.exp(-dists_sqr / (self.sigma ** 2)) / (sqrt(np.pi) * self.sigma)
        area_r = self.area_r(rr)
        pcf_r = np.sum(dists_exp) / (2 * area_r)  # 2 is there because each dist is repeated twice
        return pcf_r

    def update(self, points_, rng):
        self.points = points_
        N = len(points_)

        self.rad_values = rng
        self.pcf_values = np.empty(len(rng), dtype=np.float)
        self.get_pcf = lambda r: self.pcf_values[bisect_right(self.rad_values, r)]

        # print(len(points_))
        dists = euclidean_distances(points_)
        dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N-1)

        for ii, rr in enumerate(rng):
            pcf_r = self.pcf_r(dists_wo_diag, rr) #/ N ** 2
            self.pcf_values[ii] = pcf_r

    def compatible(self, points):
        if len(self.pcf_values) == 0:
            print('Error! First call update pcf')
            return False

        N = len(points)
        dists = euclidean_distances(points)
        dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N - 1)

        for ii, rr in enumerate(self.rad_values):
            pcf_r = self.pcf_r(dists_wo_diag, rr) #/ N ** 2

            if pcf_r > (self.pcf_values[ii] + self.compat_threshold):
                return False
        return True

    # FIXME: should work now
    def increment(self, p_new):
        dists = np.linalg.norm(self.points - p_new, axis=1)
        for ii, rr in enumerate(self.rad_values):
            pcf_r = self.pcf_r(dists, rr)  # / N ** 2
            self.pcf_values[ii] += pcf_r
        self.points = np.append(self.points, np.array(p_new).reshape(1, 2), axis=0)

    def grad(self, points_old, p_new):
        dists = np.linalg.norm(points_old - p_new, axis=1)
        grads = np.zeros((len(self.rad_values), 2), dtype=np.float)
        for ii, rr in enumerate(self.rad_values):
            grad_r = np.zeros(2, dtype=np.float)
            pcf_r_new = self.pcf_r(dists, rr)  # / N ** 2
            dtor_i = np.power(dists, 2) - rr ** 2
            for jj, p_old in enumerate(points_old):
                grad_ij = np.multiply(dtor_i[jj], 2 * (p_new - p_old))
                grad_r += grad_ij * pcf_r_new / (self.sigma ** 2) * -1
            grads[ii, :] = grad_r

        # print(grads)
        return grads

    def check_and_refine(self, points_old, p_new):
        points_old = np.array(points_old)
        grads = self.grad(points_old, p_new)

        all_points = np.append(points_old, np.array(p_new).reshape(1, 2), axis=0)
        N = len(all_points)
        dists = euclidean_distances(all_points)
        dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N - 1)

        total_grad = np.zeros((2), dtype=np.float)
        counter = 0
        for ii, rr in enumerate(self.rad_values):
            pcf_r_new = self.pcf_r(dists_wo_diag, rr)  # / N ** 2
            if pcf_r_new > (self.pcf_values[ii] + self.compat_threshold):
                total_grad += grads[ii]
                counter += 1

        alpha = 0.01
        # TODO: clip total_grad
        total_grad = total_grad / (np.linalg.norm(total_grad) + 1E-12)
        return p_new + alpha * total_grad / (counter + 1E-12)


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
    points = np.array([[0, 0], [1, 1]], dtype=np.float)
    p_to_add = np.array([2, 2], dtype=np.float)
    pcf_pattern.update(points, np.arange(0.2, 5, 0.1))
    grad_test = pcf_pattern.grad(pcf_pattern.points, p_to_add)

    # TODOL+: check
    p_to_add_refined = pcf_pattern.check_and_refine(points, p_to_add)



