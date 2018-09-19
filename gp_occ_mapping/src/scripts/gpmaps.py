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
from scipy.cluster.vq import *
import pyGPs
import copy
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PoseArray


def fusion_bcm(a, b, sa, sb):
    sz = 1. / (1. / sa + 1. / sb)
    z = sz * ((1. / sa) * a + (1. / sb) * b)
    return z, sz


class GPRMap(pyGPs.GPR):
    def __init__(self):
        super(GPRMap, self).__init__()
        self.width = 300
        self.height = 300
        self.map_size = self.width * self.height
        self.map_limit = [-15.0, 15.0, -15.0, 15.0]
        self.map_res = 0.1 #(self.map_limit[1] - self.map_limit[0]) / self.width
        self.td_res = .5

        self.map = .5 * np.ones((self.width, self.height), dtype=np.float)
        self.map_var = np.ones((self.width, self.height), dtype=np.float)

        self.x = np.zeros((1, 2))
        self.y = np.zeros((1, 1))
        self.X, self.Y = np.meshgrid(np.linspace(self.map_limit[0], self.map_limit[1], self.width),
                                     np.linspace(self.map_limit[2], self.map_limit[3], self.height))

        self.local_width = 80
        self.local_height = 80
        self.local_map_size = self.local_width * self.local_height
        self.local_map_limit = [-4.0, 4.0, -4.0, 4.0]
        self.local_map = .5 * np.ones((self.local_width, self.local_height), dtype=np.float)
        self.local_map_var = np.ones((self.local_width, self.local_height), dtype=np.float)
        self.local_X, self.local_Y = np.meshgrid(
            np.linspace(self.local_map_limit[0], self.local_map_limit[1], self.local_width),
            np.linspace(self.local_map_limit[2], self.local_map_limit[3], self.local_height))
        self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T

        self.current_pose = None
        self.robot_trajectory = None
        self.meanfunc = pyGPs.mean.Zero()
        # self.covfunc = pyGPs.cov.Matern(d=5) * pyGPs.cov.RBFard(log_ell_list=[-0.17509193324973021, -0.3272345098877003], D=2,
        # log_sigma=-1.1789938908586013)
        self.covfunc = pyGPs.cov.Matern(d=7)
        self.scan = None
        self.max_range = 5.66
        self.scan_skip = 10
        self.hyp_learned = False
        self.use_offline_hyp = True
        self.off_mean_hyp = [-0.73055028462998106]
        self.off_cov_hyp = [0.642442605255366, 2.095394555506737]
        self.off_lik_hyp = [-1.4860463474674306]
        self.w = 0.
        self.prob_free = 0.4
        self.prob_occ = 0.65
        self.first_update = True
        self.ogmap = None

    def set_scan(self, scan):
        self.scan = scan

    def logistic_reg(self, gamma=3.0):
        self.w = copy.deepcopy(self.ys2.min() / self.ys2)
        prob = 1.0 / (1.0 + np.exp(-gamma * self.ym * np.sqrt(self.w)))
        self.local_map = copy.deepcopy(np.reshape(prob, (self.local_width, self.local_height)))
        self.local_map_var = copy.deepcopy(np.reshape(self.ys2, (self.local_width, self.local_height)))

    def transform2global(self):
        self.y = np.ones((1, 1))
        self.x = np.array(-1)
        for i in range(self.scan.ranges.size):
            if self.scan.ranges[i] != np.Inf and np.mod(i, self.scan_skip) == 0:
                x1 = self.current_pose[0] + self.scan.ranges[i] * np.cos(
                    self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                x2 = self.current_pose[1] + self.scan.ranges[i] * np.sin(
                    self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                if self.x.size == 1:
                    self.x = np.array([[x1, x2]])
                else:
                    self.x = np.vstack([self.x, [x1, x2]])
                    self.y = np.vstack([self.y, [1.]])

    def training_data(self):
        for i in range(self.scan.ranges.size):
            if self.scan.ranges[i] != np.Inf and np.mod(i, self.scan_skip) == 0:
                d_max = self.scan.ranges[i] - 1.5 * self.td_res
                dist = 0.0
                while dist < d_max:
                    dist += self.td_res
                    x1 = self.current_pose[0] + dist * np.cos(
                        self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                    x2 = self.current_pose[1] + dist * np.sin(
                        self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                    self.x = np.vstack([self.x, [x1, x2]])
                    self.y = np.vstack([self.y, [-1.]])

    def separate_training_data(self):
        self.y = np.ones((1, 1))
        self.x = np.array(-1)
        for i in range(self.scan.ranges.size):
            if self.scan.ranges[i] != np.Inf and np.mod(i, self.scan_skip) == 0:
                d_max = self.scan.ranges[i] - 1.5 * self.td_res
                dist = 0.0
                while dist < d_max:
                    dist += self.td_res
                    x1 = self.current_pose[0] + dist * np.cos(
                        self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                    x2 = self.current_pose[1] + dist * np.sin(
                        self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                    if self.x.size == 1:
                        self.x = np.array([[x1, x2]])
                    else:
                        self.x = np.vstack([self.x, [x1, x2]])
                        self.y = np.vstack([self.y, [-1.]])

    def update_map(self):
        iy = int(np.round(self.current_pose[0] / self.map_res) + (self.width / 2) - (self.local_width / 2))
        ix = int(np.round(self.current_pose[1] / self.map_res) + (self.height / 2) - (self.local_height / 2))
        for i in range(self.local_map.shape[0]):
            for j in range(self.local_map.shape[1]):
                if not (ix < 0 and iy < 0):
                    if ix < self.width and iy < self.height:
                        if self.first_update:
                            self.local_map_var[i, j] *= 1000
                            self.first_update = False

                        z, sz = fusion_bcm(self.local_map[i, j], self.map[i + ix, j + iy], self.local_map_var[i, j],
                                           self.map_var[i + ix, j + iy])
                        self.map[i + ix, j + iy] = copy.deepcopy(z)
                        self.map_var[i + ix, j + iy] = copy.deepcopy(sz)

    def replace_map(self):
        iy = int(np.round(self.current_pose[0] / self.map_res) + (self.width / 2) - (self.local_width / 2))
        ix = int(np.round(self.current_pose[1] / self.map_res) + (self.height / 2) - (self.local_height / 2))
        for i in range(self.local_map.shape[0]):
            for j in range(self.local_map.shape[1]):
                if not (ix < 0 and iy < 0):
                    if ix < self.width and iy < self.height:
                        self.map[i + ix, j + iy] = copy.deepcopy(self.local_map[i, j])
                        self.map_var[i + ix, j + iy] = copy.deepcopy(self.local_map_var[i, j])

    def build_map(self):
        self.transform2global()
        self.training_data()
        self.setData(self.x, self.y)
        if not self.hyp_learned:
            # self.optimize()
            self.hyp_learned = True
        else:
            self.meanfunc.hyp = self.off_mean_hyp
            self.covfunc.hyp = self.off_cov_hyp
            self.likfunc.hyp = self.off_lik_hyp
            self.getPosterior()

        self.local_X, self.local_Y = np.meshgrid(
            np.linspace(self.local_map_limit[0] + self.current_pose[0], self.local_map_limit[1] + self.current_pose[0],
                        self.local_width),
            np.linspace(self.local_map_limit[2] + self.current_pose[1], self.local_map_limit[3] + self.current_pose[1],
                        self.local_height))
        self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T

        self.predict(self.xs)
        self.logistic_reg()
        self.update_map()

    def build_occ_map(self):
        if not self.current_pose is None:
            self.transform2global()
            self.setData(self.x, self.y)
            if not self.hyp_learned:
                # self.optimize()
                self.hyp_learned = True
            else:
                self.meanfunc.hyp = self.off_mean_hyp
                self.covfunc.hyp = self.off_cov_hyp
                self.likfunc.hyp = self.off_lik_hyp
                self.getPosterior()

            self.local_X, self.local_Y = np.meshgrid(
                np.linspace(self.local_map_limit[0] + self.current_pose[0], self.local_map_limit[1] + self.current_pose[0],
                            self.local_width),
                np.linspace(self.local_map_limit[2] + self.current_pose[1], self.local_map_limit[3] + self.current_pose[1],
                            self.local_height))
            self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T

            self.predict(self.xs)
            self.logistic_reg(gamma=5.)
            self.update_map()
        else:
            return

    def build_free_map(self):
        if not self.current_pose is None:
            self.separate_training_data()
            self.setData(self.x, self.y)
            if not self.hyp_learned:
                # self.optimize()
                self.hyp_learned = True
            else:
                self.meanfunc.hyp = self.off_mean_hyp
                self.covfunc.hyp = self.off_cov_hyp
                self.likfunc.hyp = self.off_lik_hyp
                self.getPosterior()

            self.local_X, self.local_Y = np.meshgrid(
                np.linspace(self.local_map_limit[0] + self.current_pose[0], self.local_map_limit[1] + self.current_pose[0],
                            self.local_width),
                np.linspace(self.local_map_limit[2] + self.current_pose[1], self.local_map_limit[3] + self.current_pose[1],
                            self.local_height))
            self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T

            self.predict(self.xs)
            self.logistic_reg()
            self.update_map()
        else:
            return


class TwoGPsMaps():
    def __init__(self):
        self.occ_map = GPRMap()
        self.occ_map.covfunc = pyGPs.cov.Matern(d=3)
        self.free_map = GPRMap()
        self.current_pose = None
        self.map = copy.deepcopy(self.occ_map.map)
        self.map_var = copy.deepcopy(self.occ_map.map_var)
        self.frontier_map = copy.deepcopy(self.occ_map.map)
        # self.occ_map.covfunc = pyGPs.cov.Matern(d=5)
        # self.free_map.covfunc = pyGPs.cov.Matern(d=5)
        self.X = self.occ_map.X
        self.Y = self.occ_map.Y
        self.scan = None
        self.num_of_clusters = 3
        self.expl_goal = None
        self.goal_dist = np.zeros(self.num_of_clusters)
        self.frontier_thresh = 0.55
        self.beta = 4.

    def set_scan(self, scan):
        self.occ_map.set_scan(scan)
        self.free_map.set_scan(scan)
        self.scan = scan

    def build_map(self):
        self.occ_map.current_pose = self.current_pose
        self.free_map.current_pose = self.current_pose
        self.occ_map.build_occ_map()
        self.free_map.build_free_map()
        self.merge_map()

    def merge_map(self):
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                s2_occ = self.occ_map.map_var[i, j] / self.beta
                s2_free = self.free_map.map_var[i, j]
                z, sz = fusion_bcm(self.occ_map.map[i, j], self.free_map.map[i, j], s2_occ, s2_free)
                self.map[i, j] = z
                self.map_var[i, j] = sz

    def build_frontier_map(self, gamma=3.0):
        dox, doy = np.gradient(self.occ_map.map, self.occ_map.map_res)
        dmx, dmy = np.gradient(self.map, self.occ_map.map_res)
        obs_bound = np.abs(dox) + np.abs(doy)
        all_bound = np.abs(dmx) + np.abs(dmy)
        meanf = all_bound - self.beta * (obs_bound + self.occ_map.map - 0.5)
        w = copy.deepcopy(self.map_var.min() / self.map_var)
        self.frontier_map = 1.0 / (1.0 + np.exp(-gamma * meanf * np.sqrt(w)))

    def exploration_node(self):
        self.build_frontier_map()
        frontiers_msg = PoseArray()
        idf = np.nonzero(self.frontier_map > self.frontier_thresh)
        if np.size(idf) == 0:
            # print "No frontier has been detected"
            return frontiers_msg
        data = np.column_stack([self.occ_map.X[idf[0], idf[1]], self.occ_map.Y[idf[0], idf[1]]])
        try:
            self.expl_goal, idx = kmeans2(data, self.num_of_clusters)
        except:
            # print "Clustering failed; No frontier has been detected"
            # print "The experiment is done!"
            return frontiers_msg

        for i in range(self.num_of_clusters):
            self.goal_dist[i] = np.sqrt((self.expl_goal[i, 0] - self.current_pose[0]) ** 2
                                        + (self.expl_goal[i, 1] - self.current_pose[1]) ** 2)
        id_nf = np.argsort(self.goal_dist, kind='mergesort')

        temp = copy.deepcopy(self.expl_goal)
        for i in range(self.num_of_clusters):
            self.expl_goal[i, :] = temp[id_nf[i], :]

        frontiers_msg.header.stamp = rospy.Time.now()
        frontiers_msg.header.frame_id = "map"

        for i in range(self.expl_goal.shape[0]):
            frontiers_msg.poses.append(
                Pose(Point(self.expl_goal[i, 0], self.expl_goal[i, 1], 0), Quaternion(0, 0, 0, 1)))
        return frontiers_msg

    def map_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        grid_msg.info.resolution = self.occ_map.map_res
        grid_msg.info.width = self.occ_map.width
        grid_msg.info.height = self.occ_map.height

        grid_msg.info.origin = Pose(Point(self.occ_map.map_limit[0], self.occ_map.map_limit[2], 0),
                                    Quaternion(0, 0, 0, 1))

        flat_grid = copy.deepcopy(self.map.reshape((self.occ_map.map_size,)))
        for i in range(self.occ_map.map_size):
            if flat_grid[i] > self.occ_map.prob_occ:
                flat_grid[i] = 100
            elif flat_grid[i] < self.occ_map.prob_free:
                flat_grid[i] = 0
            else:
                flat_grid[i] = -1
        flat_grid = np.round(flat_grid)
        flat_grid = flat_grid.astype(int)
        grid_msg.data = list(flat_grid)
        return grid_msg

    def gp_com_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        grid_msg.info.resolution = self.occ_map.map_res
        grid_msg.info.width = self.occ_map.width
        grid_msg.info.height = self.occ_map.height

        grid_msg.info.origin = Pose(Point(self.occ_map.map_limit[0], self.occ_map.map_limit[2], 0),
                                    Quaternion(0, 0, 0, 1))

        flat_grid = copy.deepcopy(self.map.reshape((self.occ_map.map_size,))) * 100
        flat_grid = np.round(flat_grid)
        flat_grid = flat_grid.astype(int)
        grid_msg.data = list(flat_grid)
        return grid_msg

    def goal_message(self):
        """ Return a simple navigation goal. """

        goal_msg = PoseStamped()

        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"

        if not self.expl_goal is None:
            goal_msg.pose = Pose(Point(self.expl_goal[0, 0], self.expl_goal[0, 1], 0),
                                 Quaternion(0, 0, 0, 1))
        return goal_msg
