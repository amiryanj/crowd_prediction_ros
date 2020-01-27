#!/usr/bin/env python

import numpy as np
import rospy
from trajec_prediction.srv import GetPredictedTrajectories, GetPredictedTrajectoriesResponse
from prediction_baselines.const_vel import predict_multiple
from trajec_prediction.msg import Vec3Trajectory, Vec3GroupTrajectory, Vec3GroupTrajectoryArray
from geometry_msgs.msg import Vector3Stamped


def handle_prediction_request(req):
    print("Received a Request:", req)
    res = GetPredictedTrajectoriesResponse()

    req_obsvs = req.obsvs.trajectories  # N x T x 2
    N = len(req_obsvs)
    if N > 0:
        T = len(req_obsvs[0].trajectory)

        obsvs_np = np.zeros((N, T, 2), dtype=np.float)
        for ii in range(len(req_obsvs)):
            for tt in range(obsvs_np.shape[1]):
                obsvs_np[ii, tt, 0] = req_obsvs[ii].trajectory[tt].vector.x
                obsvs_np[ii, tt, 1] = req_obsvs[ii].trajectory[tt].vector.y

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



