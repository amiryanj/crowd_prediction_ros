import os
import random

import numpy as np
from crowd_prediction.crowd_synthesis.pcf import PcfPattern, DartThrowing
from sklearn.mixture import gaussian_mixture
from crowd_prediction.opentraj_tools.parser.parser_eth import ParserETH
from sklearn import cluster, datasets, mixture


class CrowdSynthesizer:
    def __init__(self):
        self.crowd = []
        self.synthetic_crowds = []

        self.pcf_range = np.arange(0.2, 10, 0.05)
        self.grid_size = (64, 64)
        self.vel_gmm_n_components = 4
        self.target_pcf = PcfPattern()
        self.heatmap_grid = np.ones(self.grid_size, dtype=np.float)
        self.vel_gmm_grid = [[mixture.GaussianMixture(n_components=self.vel_gmm_n_components, covariance_type='tied', max_iter=100)
                              for _ in range(self.grid_size[1])]
                             for j in range(self.grid_size[0])]

        self.synthesis_max_try = 500
        self.synthesis_max_pts = 10  # FIXME: calc from data
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
        for t in dataset.t_p_dict:
            self.target_pcf.update(dataset.t_p_dict[t])
            pcf_t = self.target_pcf.pcf_values
            for ii, r in enumerate(self.target_pcf.rad_values):
                all_pcfs.append(pcf_t)
        all_pcfs = np.array(all_pcfs)
        print('*****************', all_pcfs[0].shape)

    def compute_heatmap(self, dataset):
        all_points = dataset.get_all_points()
        all_points = np.array(all_points)

        # heatmap should be smoothed
        self.heatmap_grid, xedges, yedges  = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=self.grid_size)

        x_min, x_max = min(all_points[:, 0]), max(all_points[:, 0])
        y_min, y_max = min(all_points[:, 1]), max(all_points[:, 1])
        # FIXME: grid_size[0] and grid_size[1] may need to exchange thier place
        map_to_world_coord = lambda ind: [(ind %  self.grid_size[1] +0.5) / self.grid_size[0] * (x_max - x_min) + x_min,
                                          (ind // self.grid_size[1] +0.5) / self.grid_size[1] * (y_max - y_min) + y_min]
        self.dart_thrower.set_pdf_grid(self.heatmap_grid, map_to_world_coord)

    def compute_vel_gmm(self, dataset):
        all_trajs = dataset.get_all_trajs()
        all_vs = [traj_i[1:] - traj_i[:-1] for traj_i in all_trajs]
        all_pvs = []
        for ii in range(len(all_trajs)):
            all_pvs.extend(np.stack(all_trajs[ii][:-1], all_vs[ii]))
        self.vel_gmm_grid = np.zeros(self.grid_size, dtype=float)

    # ===================================
    # ======== Synthesize Crowd =========
    # ===================================
    def synthesize_init(self, detections):
        final_points = detections.copy()

        try_counter = 0
        while try_counter < self.synthesis_max_try:
            try_counter += 1

            temp_points = final_points.copy()
            p_new = self.draw_point()
            temp_points.append(p_new)
            if self.check_pcf_error(temp_points):
                final_points.append(p_new)
            if len(final_points) == self.synthesis_max_pts:
                break

        for ii in range(len(final_points)):
            v_new = self.draw_vel(final_points[ii])
            final_points.append(v_new)
        return final_points

    def draw_point(self):
        return self.dart_thrower.random_sample()

    def draw_vel(self, p):
        # TODO: use self.gaussian_mixture
        return [0, 0]

    def check_pcf_error(self, points):
        return self.target_pcf.compatible(points)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    crowd_syn = CrowdSynthesizer()

    # TODO:
    #  n_configs = 10

    # TODO: Test
    #  rand_t = Pick a random frame
    #  all_detections = D(rand_t)
    #  if len(all_detections) > 4:
    #       partial_detection = all_detections[:3]
    #       syn_crowds = [None] * n_configs
    #       for kk in range(n_configs):
    #           syn_crowds[kk] = crowd_syn.synthesize_init(partial_detection)


    parser = ParserETH()
    dir = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth'
    parser.load(os.path.join(dir, 'obsmat.txt'))
    crowd_syn.extract_features(parser)

    all_ts = parser.t_p_dict.keys()
    t_begin, t_end = min(all_ts), max(all_ts)
    all_ps = parser.get_all_points()
    all_ps = np.array(all_ps)
    x_min, x_max = min(all_ps[:, 0]), max(all_ps[:, 0])
    y_min, y_max = min(all_ps[:, 1]), max(all_ps[:, 1])

    for kk in range(100):
        (rnd_t, gt_pts) = random.choice(list(parser.t_p_dict.items()))
        gt_pts = np.array(gt_pts)
        if len(gt_pts) < 4: continue
        final_pnts = crowd_syn.synthesize_init(gt_pts[:3].tolist())
        final_pnts = np.array(final_pnts)

        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], color='green', s=100)
        plt.scatter(final_pnts[:, 0], final_pnts[:, 1], marker='o')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        # plt.ion()
        plt.show()
        # plt.pause(10.0)

        print('here')



