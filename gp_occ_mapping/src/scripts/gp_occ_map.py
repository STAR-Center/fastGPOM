#!/usr/bin/env python

""" Copyright (C) 2016  Maani Ghaffari Jadidi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details."""

import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib
import gpmaps
import rospy
import rosbag

from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PoseStamped, PoseArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from nav_msgs.srv import GetMap
from tf.transformations import euler_from_quaternion


def pose_update_callback(slam_out_pose):
    q = [slam_out_pose.pose.orientation.x, slam_out_pose.pose.orientation.y, slam_out_pose.pose.orientation.z,
         slam_out_pose.pose.orientation.w]
    angles = euler_from_quaternion(q)
    pose = np.array([slam_out_pose.pose.position.x, slam_out_pose.pose.position.y, angles[2]])
    gp_map.current_pose = pose


def laser_scan_callback(scan):
    scan.ranges = np.asarray(scan.ranges)
    gp_map.set_scan(scan)


def listener():
    rospy.Subscriber("slam_out_pose", numpy_msg(PoseStamped), pose_update_callback)
    rospy.Subscriber("scan", numpy_msg(LaserScan), laser_scan_callback)
    rospy.spin()


def occ_map_build_callback():
    global gp_com_msg
    while not rospy.is_shutdown():
        if not gp_map.scan is None and not gp_map.current_pose is None:
            gp_map.build_map()
            front_msg = gp_map.exploration_node()

            occ_map_msg = gp_map.map_message()
            gp_com_msg = gp_map.gp_com_message()
            gp_com_pub.publish(gp_com_msg)
            map_pub.publish(occ_map_msg)
            map_data_pub.publish(occ_map_msg.info)
            front_pub.publish(front_msg)

            goal_msg = gp_map.goal_message()
            goal_pub.publish(goal_msg)

            plot_current_map()


def get_map_callback(req):
    global gp_com_msg
    rospy.loginfo("Receiving GP occupancy map request.")
    return gp_com_msg


def occ_map_build():
    global gp_com_msg

    # reading dataset(bag file)
    file_name = '/home/kuang/software/Robotics/mcmc/catkin_ws/src/fun-gpom/data_set/stdr_data.bag'
    bag = rosbag.Bag(file_name)

    # We want to get scan and pose, so we should do mapping after 2 iterations.
    mapping_flag = False

    # 2 msgs
    scan_msg = None
    pose_msg = None

    count = 0

    for topic, msg, t in bag.read_messages(topics=['/slam_out_pose', '/robot0/laser_0']):
        if topic == '/robot0/laser_0':
            scan_msg = msg
            scan_msg.ranges = np.asarray(scan_msg.ranges)
            gp_map.set_scan(scan_msg)
            continue
        elif topic == '/slam_out_pose':
            pose_msg = msg
            q = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z,
                 pose_msg.pose.orientation.w]
            angles = euler_from_quaternion(q)
            pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, angles[2]])
            gp_map.current_pose = pose

        if not gp_map.scan is None and not gp_map.current_pose is None:
            gp_map.build_map()
            front_msg = gp_map.exploration_node()

            occ_map_msg = gp_map.map_message()
            gp_com_msg = gp_map.gp_com_message()
            gp_com_pub.publish(gp_com_msg)
            map_pub.publish(occ_map_msg)
            map_data_pub.publish(occ_map_msg.info)
            front_pub.publish(front_msg)

            goal_msg = gp_map.goal_message()
            goal_pub.publish(goal_msg)

            plot_current_map()

            print(count)
            count += 1

def plot_current_map():
    font = {'weight': 'normal',
            'size': 20}

    matplotlib.rc('font', **font)

    plt.figure("GP Occupancy Map")
    plt.clf()
    plt.pcolor(gp_map.X, gp_map.Y, gp_map.map, vmin=0, vmax=1)
    plt.colorbar()
    if not gp_map.current_pose is None:
        plt.quiver(gp_map.current_pose[0], gp_map.current_pose[1], 1. * np.cos(gp_map.current_pose[2]),
                   1. * np.sin(gp_map.current_pose[2]), angles='xy', scale_units='xy', scale=1,
                   edgecolors='m', pivot='mid', facecolor='none', linewidth=1, width=0.001, headwidth=400, headlength=800)
    plt.axis('equal')

    plt.figure("GP Frontier Map")
    plt.clf()
    plt.pcolor(gp_map.X, gp_map.Y, gp_map.frontier_map, vmin=0, vmax=1)
    plt.quiver(gp_map.current_pose[0], gp_map.current_pose[1], 1. * np.cos(gp_map.current_pose[2]),
               1. * np.sin(gp_map.current_pose[2]), angles='xy', scale_units='xy', scale=1,
               edgecolors='m', pivot='mid', facecolor='none', linewidth=1, width=0.001, headwidth=400, headlength=800)
    if not gp_map.expl_goal is None:
        plt.plot(gp_map.expl_goal[:, 0], gp_map.expl_goal[:, 1], linestyle='-.', c='m', marker='+', markersize=14)
    plt.axis('equal')
    plt.draw()
    plt.pause(.1)


if __name__ == '__main__':
    try:
        gp_map = gpmaps.TwoGPsMaps()
        plt.ion(), plt.show()

        rospy.init_node('gp_occ_map', anonymous=True)

        gp_com_msg = gp_map.gp_com_message()
        gp_com_pub = rospy.Publisher('gp_com', OccupancyGrid, queue_size=10, latch=True)
        map_pub = rospy.Publisher('gp_map', OccupancyGrid, queue_size=10, latch=True)
        map_data_pub = rospy.Publisher('gp_map_metadata', MapMetaData, queue_size=10, latch=True)
        goal_pub = rospy.Publisher('gp_goal', PoseStamped, queue_size=10, latch=True)
        front_pub = rospy.Publisher('gp_frontiers', PoseArray, queue_size=10, latch=True)
        s = rospy.Service('gp_map_server', GetMap, get_map_callback)

        t_lis = threading.Thread(name='gp_map_listener', target=listener)
        t_map = threading.Thread(name='gp_map_building', target=occ_map_build_callback)
        t_lis.start()
        t_map.start()

        # occ_map_build()

    except rospy.ROSInterruptException:
        pass
