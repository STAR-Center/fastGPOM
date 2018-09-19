#!/usr/bin/env python

import rospy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def callback(odom):
    header = odom.header
    pose = odom.pose.pose

    ps = PoseStamped()
    ps.header = header
    ps.pose = pose

    posestamped_pub = rospy.Publisher('/slam_out_pose', PoseStamped, queue_size=10)

    posestamped_pub.publish(ps)


if __name__ == '__main__':
    rospy.init_node('odom_to_posestamped', anonymous=True)

    rospy.Subscriber('/odometry/filtered', Odometry, callback)

    rospy.spin()