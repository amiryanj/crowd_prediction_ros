#!/usr/bin/env python

import numpy as np
import sys
import rospy
from trajec_prediction.msg import Vec3Trajectory, Vec3GroupTrajectory, Vec3GroupTrajectoryArray
from geometry_msgs.msg import Vector3Stamped
from trajec_prediction.srv import GetPredictedTrajectories, GetPredictedTrajectoriesResponse


def trajec_prediction_client(obsvs, n_next, n_samples=1):
    rospy.wait_for_service('trajec_prediction')
    try:
        trajec_prediction = rospy.ServiceProxy('trajec_prediction', GetPredictedTrajectories)
        obsvs_msg = Vec3GroupTrajectory()
        for ii in range(obsvs.shape[0]):
            obsv_trajec = Vec3Trajectory()
            for tt in range(obsvs.shape[1]):
                obsv_loc = Vector3Stamped()
                obsv_loc.vector.x = obsvs[ii, tt, 0]
                obsv_loc.vector.y = obsvs[ii, tt, 1]
                obsv_loc.vector.z = 0
                obsv_loc.header.seq = tt  # FIXME: put the right time stamp here
                obsv_trajec.trajectory.append(obsv_loc)
            obsvs_msg.trajectories.append(obsv_trajec)
            print(obsvs_msg)
        res = trajec_prediction(obsvs_msg, n_next, n_samples)
        return res

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


if __name__ == "__main__":
    print("Calling Prediction Client")
    obsv = np.array([[i, i] for i in range(3)]).astype(np.float).reshape((1, -1, 2))
    # obsv = np.array([[0, 0], [1, 1], [2, 2], [3, 3]]).astype(np.float).reshape((1, -1, 2))
    print(obsv.shape)

    preds = trajec_prediction_client(obsv, 4, 1)
    print(preds)


