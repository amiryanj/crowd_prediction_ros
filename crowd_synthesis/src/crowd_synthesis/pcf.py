import os
import numpy as np
from math import sqrt
from sklearn.metrics.pairwise import euclidean_distances

# from followbot.temp.parser_eth import ParserETH


def pcf(points, rng=np.arange(0.2, 10, 0.05), sigma=0.3):
    N = len(points)
    dists = euclidean_distances(points)
    dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N-1)
    # return dists

    pcf_lut = dict()  # PCF look-up table
    for rr in rng:
        A_r = np.pi * (rr + 0.2) ** 2 - np.pi * max((rr - 0.2), 0) ** 2

        dists_sqr = np.power(dists_wo_diag - rr, 2)
        dists_exp = np.exp(-dists_sqr / (sigma ** 2)) / (sqrt(np.pi) * sigma)
        pcf_r = np.sum(dists_exp) / N**2 / 2. /A_r # because each dist is repeated twice

        pcf_lut[rr] = pcf_r

    return pcf_lut


# ===========================
# === Synthesis Algorithm ===
# ===========================
def sample_pom(pdf, real_points):
    return np.random.rand(2)


def dart_with_social_space(points, thresh=0.5):
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


def dart_throwing(n, target_pcf, prior_pom):
    error_eps = 0.5
    points = []
    counter = 0
    while len(points) < n:
        new_pt = sample_pom(pdf=prior_pom)
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
    crowd_by_click()





