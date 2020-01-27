#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan  # TODO: which message we will use?
from baselines.const_vel import predict_single, predict_multiple
from std_msgs.msg import String


def run_sim():
    # TODO: read the topic from rospy.load_param('~xxx_topic')
    global pub
    pub = rospy.Publisher('/crowd_prediction/pred_trajecs', LaserScan, queue_size=1)
    sub = rospy.Subscriber("/crowd_syn/all_agents", String, robot_nav_callback)
    while not rospy.is_shutdown():
        msg = LaserScan()


def robot_nav_callback(data):
    msg = String()
    preds = predict_multiple(data)
    msg.data = preds
    pub.publish(msg)


if __name__ == '__main__':
    try:
        run_sim()
    except rospy.ROSInterruptException:
        pass

