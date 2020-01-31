import os
import numpy as np
from crowd_prediction.crowd_synthesis.pcf import pcf
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
        self.pcf_map = np.zeros(self.pcf_range.size)
        self.heatmap_grid = np.zeros(self.grid_size, dtype=np.float)
        self.vel_gmm_grid = [[mixture.GaussianMixture(n_components=self.vel_gmm_n_components, covariance_type='tied', max_iter=100)
                               for _ in range(self.grid_size[1])]
                                for j in range(self.grid_size[0])]

        self.synthesis_stop_n_try = 500

    # ===================================
    # ======== Extract Features =========
    # ===================================
    def extract_features(self, dataset):
        self.pcf_map = self.compute_avg_pcf(dataset)
        self.heatmap_grid = self.compute_heatmap(dataset)
        self.vel_gmm_grid = self.compute_vel_gmm(dataset)

    def compute_avg_pcf(self, dataset):
        all_pcfs = []
        for t in dataset.t_p_dict:
            pcf_t = pcf(dataset.t_p_data[t])
            all_pcfs.append(pcf_t)
        avg_pcf = np.array(all_pcfs).mean(axis=1)
        return avg_pcf

    def compute_heatmap(self, dataset):
        all_points = dataset.get_all_points()

        all_points = np.array(all_points)
        heatmap_grid = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=50)
        return heatmap_grid


    def compute_vel_gmm(self, dataset):
        all_trajs = dataset.get_all_trajs()
        all_pv =
        pass

    # ===================================
    # ======== Synthesize Crowd =========
    # ===================================
    def synthesize_init(self, detections):
        temp_points = []
        final_points = detections.copy()

        try_counter = 0
        while try_counter < self.synthesis_stop_n_try:
            try_counter += 1

            temp_points = final_points.copy()
            p_new = self.draw_point()
            temp_points.append(p_new)
            if self.check_pcf_error(temp_points):
                final_points.append(p_new)

        for ii in range(len(final_points)):
            v_new = self.draw_vel(final_points[ii])
            final_points.append(v_new)

    def draw_point(self):
        return [0, 0]

    def draw_vel(self, p):
        return [0, 0]

    def check_pcf_error(self, points):
        return True


if __name__ == '__main__':
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



