#!/usr/bin/env python
import numpy as np

import rospy
from crowd_prediction.prediction_baselines.const_vel import predict_single, predict_multiple
from crowd_prediction.crowd_synthesis.crowd_synthesis import CrowdSynthesizer
from OpenTraj.tools.parser.parser_eth import ParserETH
from OpenTraj.tools.parser.parser_gc import ParserGC
# msgs
from trajec_prediction.msg import Vec3Trajectory, Vec3GroupTrajectory, Vec3GroupTrajectoryArray
from frame_msgs.msg import DetectedPerson, DetectedPersons, TrackedPersons
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Header


class SimplePrediction:
    def __init__(self):
        print('started prediction node')
        self.tracks = dict()
        self.cur_t = -1

        self.crowd_syn = CrowdSynthesizer
        # dataset = ParserGC('/home/cyrus/workspace2/OpenTraj/GC/Annotation')
        dataset = ParserETH('/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/obsmat.txt')
        self.crowd_syn.extract_features(dataset)
        self.n_syn_configs = 10
        self.synthetic_crowd = []

        self.n_past = rospy.get_param("~Prediction/n_past")
        self.n_next = rospy.get_param("~Prediction/n_next")
        self.n_samples = rospy.get_param("~Prediction/n_samples")

        # tracking_topic = rospy.get_param("~subscriber/")
        tracking_topic = "/rwth_tracker/tracked_persons"
        traj_prediction_topic = rospy.get_param("~publisher/trajec_pred/topic")

        self.sub = rospy.Subscriber(tracking_topic, TrackedPersons, self.callback_tracking)
        self.pub = rospy.Publisher(traj_prediction_topic, Vec3GroupTrajectoryArray, queue_size=1)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def callback_tracking(self, track_msg):
        # FIXME: call me regularly
        if self.cur_t == -1:  #  just first time
            for kk in range(self.n_syn_configs):
                self.synthetic_crowd = []  # TODO

        last_ids = []
        for track in track_msg.tracks:
            if not track.track_id in self.tracks:
                self.tracks[track.track_id] = []
            self.tracks[track.track_id].append([track.pose.pose.position.x, track.pose.pose.position.y])
            last_ids.append(track.track_id)

        for id in self.tracks.keys():
            if id not in last_ids:
                del self.tracks[id]

        self.cur_t = track_msg.header.seq

        ped_obsvs = []
        ped_ids = []
        msg = Vec3GroupTrajectoryArray()

        for id, track in self.tracks.items():
            if len(track) > self.n_past:
                ped_obsvs.append(track[len(track)-self.n_past:])
                ped_ids.append(id)
        if ped_obsvs:
            ped_obsvs = np.stack(ped_obsvs)
            print('ped_obsvs.shape = ', ped_obsvs.shape)
            for _ in range(self.n_samples):
                sample_k = Vec3GroupTrajectory()
                preds = predict_multiple(ped_obsvs, self.n_next)
                print('ped_ids = ', ped_ids, preds.shape)
                for i, pred_i in enumerate(preds):
                    pred_i_msg = Vec3Trajectory()
                    pred_i_msg.id = ped_ids[i]
                    for t, p_t in enumerate(pred_i):
                        p_i_t = Vector3Stamped()
                        p_i_t.vector.x = p_t[0]
                        p_i_t.vector.y = p_t[1]
                        p_i_t.vector.z = 0
                        p_i_t.header.frame_id = ""
                        p_i_t.header.seq = self.cur_t + t + 1

                        pred_i_msg.trajectory.append(p_i_t)
                    sample_k.trajectories.append(pred_i_msg)
                msg.trajectory_samples.append(sample_k)
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('simple_prediction')
    try:
        SimplePrediction()
    except rospy.ROSInterruptException:
        pass

