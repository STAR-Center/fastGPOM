#!/usr/bin/env python
'''
this code is for online_gpom
'''
import gpom as mcgpom
import rospy
import rosbag
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PoseArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from nav_msgs.srv import GetMap
from tf.transformations import euler_from_quaternion

import time


# 1. for each frame, build the local gaussian process map [mu, sigma, prob]
# 1.1 read one frame, get current posi, ori, x,y

# 1.2 gaussian process
# 1.2.1 for the first frame, do maximum likelyhood to estimate hyperparam

# 1.2.2 predict on top of kernel. achieve mu and sigma

# 1.2.3 logistic regression to calculate prob

# 1.3. fuse two frame whenever update

def plot_current_map():
    font = {'weight': 'normal',
            'size': 20}

    matplotlib.rc('font', **font)

    plt.figure("GP Occupancy Map")
    plt.clf()
    plt.pcolor(gp_map.X, gp_map.Y, gp_map.map, vmin=0, vmax=1)
    plt.colorbar()
    '''
    if not gp_map.current_pose is None:
        plt.quiver(gp_map.current_pose[0], gp_map.current_pose[1], 1. * np.cos(gp_map.current_pose[2]),
                   1. * np.sin(gp_map.current_pose[2]), angles='xy', scale_units='xy', scale=1,
                   edgecolors='m', pivot='mid', facecolor='none', linewidth=1, width=0.001, headwidth=400, headlength=800)
    plt.axis('equal')
    '''
    '''
    plt.figure("GP Frontier Map")
    plt.clf()
    plt.pcolor(gp_map.X, gp_map.Y, gp_map.frontier_map, vmin=0, vmax=1)
    plt.quiver(gp_map.current_pose[0], gp_map.current_pose[1], 1. * np.cos(gp_map.current_pose[2]),
               1. * np.sin(gp_map.current_pose[2]), angles='xy', scale_units='xy', scale=1,
               edgecolors='m', pivot='mid', facecolor='none', linewidth=1, width=0.001, headwidth=400, headlength=800)
    if not gp_map.expl_goal is None:
        plt.plot(gp_map.expl_goal[:, 0], gp_map.expl_goal[:, 1], linestyle='-.', c='m', marker='+', markersize=14)
    plt.axis('equal')
    '''
    plt.draw()
    plt.pause(.1)


def publish_map_image():
    grid_msg = OccupancyGrid()
    grid_msg.header.stamp = rospy.Time.now()
    grid_msg.header.frame_id = "map"

    grid_msg.info.resolution = gp_map.map_res
    grid_msg.info.width = gp_map.width
    grid_msg.info.height = gp_map.height

    grid_msg.info.origin = Pose(Point(gp_map.map_limit[0], gp_map.map_limit[2], 0),
                                Quaternion(0, 0, 0, 1))

    flat_grid = gp_map.map.copy()
    flat_grid = flat_grid.reshape((gp_map.map_size,))

    flat_grid[np.where(flat_grid<0.4)] = 0
    flat_grid[np.where(flat_grid > 0.65)] = -100
    flat_grid[np.where(flat_grid > 0.4)] = 1

    #flat_grid = gp_map.threshold(flat_grid)

    flat_grid = -flat_grid#np.round(-flat_grid)
    flat_grid = flat_grid.astype(int)

    grid_msg.data = flat_grid.tolist()

    occ_map_pub.publish(grid_msg)


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

            publish_map_image()

            # plot_current_map()


if __name__ == '__main__':
    try:
        gp_map = mcgpom.GPRMap(mcmc=False)
        rospy.init_node('gp_occ_map', anonymous=True)
        '''
        #publisher
        map_pub = rospy.Publisher('gp_map', OccupancyGrid, queue_size=10, latch=True)
        map_data_pub = rospy.Publisher('gp_map_metadata', MapMetaData, queue_size=10, latch=True)
        goal_pub = rospy.Publisher('gp_goal', PoseStamped, queue_size=10, latch=True)
        front_pub = rospy.Publisher('gp_frontiers', PoseArray, queue_size=10, latch=True)
        #srv
        s = rospy.Service('gp_map_server', GetMap, get_map_callback)
        '''

        # publish map
        occ_map_pub = rospy.Publisher('gp_occ_map', OccupancyGrid, queue_size=10, latch=True)

        # publish
        '''
        occ_map_msg = gp_map.map_message()
        map_pub.publish(occ_map_msg)
        map_data_pub.publish(occ_map_msg.info)
    
        goal_msg = gp_map.goal_message()
        goal_pub.publish(goal_msg)
        '''

        t_lis = threading.Thread(name='gp_map_listener', target=listener)
        t_map = threading.Thread(name='gp_map_building', target=occ_map_build_callback)
        t_lis.start()
        t_map.start()
    except rospy.ROSInterruptException:
        pass
