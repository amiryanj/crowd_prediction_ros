#!/usr/bin/env python

import numpy as np
import rospy
from crowd_synthesis.srv import GetPredictedTrajectories, GetPredictedTrajectoriesResponse
from crowd_synthesis.msg import MovingParticleStamped, ParticleCrowd, ParticleCrowdArray
from geometry_msgs.msg import Vector3Stamped
from crowd_synthesis.crowd_synthesis import CrowdSynthesize


def handle_synthesis_request(req):
    print("Received a Request:", req)
    req_detections = req.detections  # N x MovingParticle[p, v, t, id]
    # n_samples = req.n_samples
    res = GetSynthesizedCrowdResponse()

    N = len(req_detections.particles)
    detections_np = []
    if N > 0:
        for ii in range(N):
            detections_np.append([req_detections.particles[ii].position.x,
                                  req_detections.particles[ii].position.y,
                                  req_detections.particles[ii].velocity.x,
                                  req_detections.particles[ii].velocity.y])
        detections_np = np.array(detections_np, dtype=np.float)

        # TODO: run synthesizer (detections_np, req.n_samples)  => synthetic_peds
        synthetic_peds = CrowdSynthesize.synthesize(detections_np, req.n_samples)

        for kk, config_kk in enumerate(synthetic_peds):
            res.configs.append(ParticleCrowd())
            for ii in range(len(synthetic_peds)):
                syn_particle = MovingParticleStamped()
                syn_particle.position.x = config_kk[ii][0]
                syn_particle.position.y = config_kk[ii][1]
                syn_particle.velocity.x = config_kk[ii][2]
                syn_particle.velocity.y = config_kk[ii][3]
                res.configs[kk].particles.append(syn_particle)
    return res


def trajec_prediction_server():
    rospy.init_node('trajec_prediction_server')
    s = rospy.Service('trajec_prediction', GetPredictedTrajectories, handle_prediction_request)
    print("Prediction Service is ready")
    rospy.spin()


if __name__ == "__main__":
    trajec_prediction_server()
