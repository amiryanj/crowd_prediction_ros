import os
import random
import numpy as np
from sklearn.mixture import gaussian_mixture
from sklearn import cluster, datasets, mixture
from bisect import bisect_left, bisect_right

from crowd_prediction.crowd_synthesis.pcf import PcfPattern, DartThrowing
from OpenTraj.tools.parser.parser_eth import ParserETH
from OpenTraj.tools.parser.parser_gc import ParserGC


class CustomDistribution:
    def __init__(self):
        self.points = []

    def fit(self, X):
        X = np.array(X)
        self.points = X

    def sample(self, n_samples=1):
        ind = 0
        if len(self.points) > 1:
            ind = np.random.randint(0, len(self.points))
        return [self.points[ind] + np.random.randn(2) * 0.1], n_samples


class CrowdSynthesizer:
    def __init__(self):
        self.crowd = []
        self.synthetic_crowds = []

        self.pcf_range = np.arange(0.2, 5, 0.1)
        self.grid_size = (64, 64)
        self.vel_gmm_n_components = 3
        self.average_pcf = PcfPattern()
        self.heatmap_grid = np.ones(self.grid_size, dtype=np.float)
        self.vel_pdf_grid = [[None for i in range(self.grid_size[1])]
                             for j in range(self.grid_size[0])]
        self.map_to_grid_coord = lambda x, y: (0, 0)

        self.synthesis_max_try = 5000
        self.n_agetns_histogram = np.empty(100)
        self.synthesis_max_pts = 100  # FIXME: calc from data
        self.dart_thrower = DartThrowing()

    # ===================================
    # ======== Extract Features =========
    # ===================================
    def extract_features(self, dataset):
        self.compute_target_pcf(dataset)
        self.compute_heatmap(dataset)
        # self.compute_vel_gmm(dataset)

    def compute_target_pcf(self, dataset):
        all_pcfs = []
        self.n_agetns_histogram *= 0

        for t in dataset.t_p_dict:
            self.average_pcf.update(dataset.t_p_dict[t], self.pcf_range)
            pcf_t = self.average_pcf.pcf_values
            all_pcfs.append(pcf_t)

            N = len(dataset.t_p_dict[t])
            if N < len(self.n_agetns_histogram):
                self.n_agetns_histogram[N] += 1
        self.average_pcf.pcf_values = np.array(all_pcfs).mean(axis=0)

    def compute_heatmap(self, dataset):
        all_points = dataset.get_all_points()
        all_points = np.array(all_points)

        # heatmap should be smoothed
        hist, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=self.grid_size)
        self.heatmap_grid = hist.T

        x_min, x_max = min(all_points[:, 0]), max(all_points[:, 0])
        y_min, y_max = min(all_points[:, 1]), max(all_points[:, 1])
        # FIXME: grid_size[0] and grid_size[1] may need to exchange thier place
        map_to_world_coord = lambda ind: [(ind %  self.grid_size[1] +0.5) / self.grid_size[0] * (x_max - x_min) + x_min,
                                          (ind // self.grid_size[1] +0.5) / self.grid_size[1] * (y_max - y_min) + y_min]
        self.map_to_grid_coord = lambda x, y: [int(np.floor((x-x_min)/(x_max-x_min+1E-6) * self.grid_size[0])),
                                               int(np.floor((y-y_min)/(y_max-y_min+1E-6) * self.grid_size[1]))]
        self.dart_thrower.set_pdf_grid(self.heatmap_grid, map_to_world_coord)

    def compute_vel_gmm(self, dataset):
        all_trajs = dataset.get_all_trajs()
        all_vs = [traj_i[1:] - traj_i[:-1] for traj_i in all_trajs]
        all_pvs = []
        grid_vs = [ [[] for ii in range(self.grid_size[1])]
                    for jj in range(self.grid_size[0])]
        for ii in range(len(all_trajs)):
            pvs_i = np.concatenate([all_trajs[ii][:-1], all_vs[ii]], axis=1)
            all_pvs.extend(pvs_i)
        for pv in all_pvs:
            u, v = self.map_to_grid_coord(pv[0], pv[1])
            grid_vs[u][v].append(pv[2:])

        for u in range(self.grid_size[0]):
            for v in range(self.grid_size[1]):
                # print(grid_vs[u][v])
                # print('variance = ', var_uv)

                if len(grid_vs[u][v]) == 0:
                    grid_vs[u][v] = [np.array([0, 0], dtype=np.float)]

                var_uv = np.var(grid_vs[u][v], axis=0)
                if len(grid_vs[u][v]) > self.vel_gmm_n_components and np.linalg.norm(var_uv) > 1E-6:
                    self.vel_pdf_grid[u][v] = mixture.GaussianMixture(n_components=2, covariance_type='diag', max_iter=10)
                else:
                    self.vel_pdf_grid[u][v] = CustomDistribution()

                self.vel_pdf_grid[u][v].fit(grid_vs[u][v])

    # ===================================
    # ======== Synthesize Crowd =========
    # ===================================
    def synthesize_init(self, detections):
        final_points = detections.copy()
        target_pcf = PcfPattern()
        target_pcf.update(detections, self.pcf_range)
        target_pcf.pcf_values = np.maximum(self.average_pcf.pcf_values, target_pcf.pcf_values)

        rnd_value = np.random.rand()
        weights = self.n_agetns_histogram.cumsum() / self.n_agetns_histogram.sum()
        max_n_points = bisect_left(weights, rnd_value)
        # max_n_points = self.synthesis_max_pts

        try_counter = 0
        while try_counter < self.synthesis_max_try:
            try_counter += 1

            temp_points = final_points.copy()
            p_new = self.draw_point()
            temp_points.append(p_new)

            if target_pcf.compatible(temp_points):
                final_points.append(p_new)

                # FIXME: add the velocity vector
                # v_new = self.draw_vel(p_new)
                # final_points[-1] = np.append(p_new, v_new)

            # FIXME: refinement does not work
            # else:
            #     temp_points.pop()
            #     p_new_new = target_pcf.check_and_refine(temp_points, p_new)
            #     temp_points.append(p_new_new)
            #     if target_pcf.compatible(temp_points):
            #         final_points.append(p_new_new)
            #         print('Refinement was successful!')

            if len(final_points) >= max_n_points:
                break


        return final_points

    def draw_point(self):
        return self.dart_thrower.random_sample()

    def draw_vel(self, p):
        # TODO: use self.gaussian_mixture
        u, v = self.map_to_grid_coord(p[0], p[1])
        s_pnt, _ = self.vel_pdf_grid[u][v].sample()
        return s_pnt


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    crowd_syn = CrowdSynthesizer()

    # TODO: Algorithm
    #  rand_t = Pick a random frame
    #  all_detections = D(rand_t)
    #  if len(all_detections) > 4:
    #       partial_detection = all_detections[:3]
    #       syn_crowds = [None] * n_configs
    #       for kk in range(n_configs):
    #           syn_crowds[kk] = crowd_syn.synthesize_init(partial_detection)

    # parser = ParserETH('/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/obsmat.txt')
    dataset = ParserGC('/home/cyrus/workspace2/OpenTraj/GC/Annotation')
    crowd_syn.extract_features(dataset)

    all_ts = dataset.t_p_dict.keys()
    t_begin, t_end = min(all_ts), max(all_ts)
    all_ps = dataset.get_all_points()
    all_ps = np.array(all_ps)
    x_min, x_max = min(all_ps[:, 0]), max(all_ps[:, 0])
    y_min, y_max = min(all_ps[:, 1]), max(all_ps[:, 1])

    for kk in range(100):
        (rnd_t, gt_pts) = random.choice(list(dataset.t_p_dict.items()))
        gt_pts = np.array(gt_pts)
        if len(gt_pts) >= 4:
            break

    # TODO:
    n_configs = 10
    n_gt_points = len(gt_pts)
    keep_k = n_gt_points // 5

    for kk in range(n_configs):
        final_pnts = crowd_syn.synthesize_init(gt_pts[:keep_k].tolist())
        final_pnts = np.array(final_pnts)

        # plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(np.flipud(crowd_syn.heatmap_grid))
        plt.title("Heatmap")

        # plt.figure()
        plt.subplot(1, 3, 2)
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], color='green', s=100, label='ground truth points')
        plt.scatter(gt_pts[:keep_k, 0], gt_pts[:keep_k, 1], color='yellow', s=80, label='given points')
        plt.scatter(final_pnts[:, 0], final_pnts[:, 1], color='red', marker='x', label='syn')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.legend()
        plt.title("frame = %d" % rnd_t)
        # plt.ion()
        # plt.pause(10.0)

        pcf_gt = PcfPattern()
        pcf_gt.update(gt_pts, crowd_syn.pcf_range)

        pcf_final = PcfPattern()
        pcf_final.update(final_pnts, crowd_syn.pcf_range)

        pcf_start = PcfPattern()
        pcf_start.update(gt_pts[:3], crowd_syn.pcf_range)

        # plt.figure()
        plt.subplot(1, 3, 3)
        plt.title("PCF")
        plt.plot(crowd_syn.average_pcf.rad_values, crowd_syn.average_pcf.pcf_values, label='Target')
        plt.plot(pcf_gt.rad_values, pcf_gt.pcf_values, label='Ground Truth')
        plt.plot(pcf_start.rad_values, pcf_start.pcf_values, label='Initial Staet')
        plt.plot(pcf_final.rad_values, pcf_final.pcf_values, label='Final State')
        plt.legend()


        # plt.ylim([0, 0.15])
        plt.show()



