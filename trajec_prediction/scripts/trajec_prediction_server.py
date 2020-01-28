#!/usr/bin/env python

import numpy as np
import rospy
from trajec_prediction.srv import GetSynthesizedCrowd
from trajec_prediction.msg import Vec3Trajectory, Vec3GroupTrajectory, Vec3GroupTrajectoryArray
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header


def handle_synthesis_request(req):
    print("Received a Request:", req)
    req_detections = req.detections  # N x MovingParticle[p, v, t, id]
    # n_samples = req.n_samples
    res = GetSynthesizedCrowdResponse()

    N = len(req_detections)
    poss = []
    vels = []
    if N > 0:
        for ii in range(N):
            poss.append(req_detections)

        preds = predict_multiple(obsvs_np, req.n_next, req.n_samples)  # 1 x N x T x 2
        preds = preds.reshape((req.n_samples, N, req.n_next, 2))

        for kk in range(req.n_samples):
            group_trjecs = Vec3GroupTrajectory()
            for ii in range(preds.shape[1]):
                pred_msg = Vec3Trajectory()
                for tt in range(req.n_next):
                    loc_i_t = Vector3Stamped()
                    loc_i_t.header.seq = tt + 1  # FIXME
                    loc_i_t.vector.x = preds[kk, ii, tt, 0]
                    loc_i_t.vector.y = preds[kk, ii, tt, 1]
                    loc_i_t.vector.z = 0
                    pred_msg.trajectory.append(loc_i_t)
                group_trjecs.trajectories.append(pred_msg)
            res.predss.trajectory_samples.append(group_trjecs)
    return res


def trajec_prediction_server():
    rospy.init_node('trajec_prediction_server')
    s = rospy.Service('trajec_prediction', GetPredictedTrajectories, handle_prediction_request)
    print("Prediction Service is ready")
    rospy.spin()


if __name__ == "__main__":
    trajec_prediction_server()



