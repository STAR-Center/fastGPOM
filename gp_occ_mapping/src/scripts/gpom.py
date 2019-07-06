
import numpy as np
from scipy.cluster.vq import *
import pyGPs
#import gpflow
import copy
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PoseArray
from scipy.stats import norm
#from utils.tools import graph_in_poly
from skimage.draw import polygon
import time
import matplotlib.pyplot as plt

def fusion_bcm(a, b, sa, sb):
    sz = 1. / (1. / sa + 1. / sb)
    z = sz * ((1. / sa) * a + (1. / sb) * b)
    return z, sz



class GPRMap(pyGPs.GPR):
    def __init__(self, mcmc = False):
        super(GPRMap, self).__init__()
        #mode
        self.mcmc = mcmc
        #map property
        self.width = 300 * 5
        self.height = 300 * 5
        self.map_size = self.width * self.height
        self.map_limit = [-15.0, 15.0, -15.0, 15.0] #*2
        self.map_res = (self.map_limit[1] - self.map_limit[0]) / self.width
        self.td_res = .5
        #map
        self.map = .5 * np.ones((self.width, self.height), dtype=np.float)
        self.map_mu = .5 * np.ones((self.width, self.height), dtype=np.float)
        self.map_sigma = np.ones((self.width, self.height), dtype=np.float)
        #data
        self.x = np.zeros((1, 2))
        self.y = np.zeros((1, 1))
        self.X, self.Y = np.meshgrid(np.linspace(self.map_limit[0], self.map_limit[1], self.width),
                                     np.linspace(self.map_limit[2], self.map_limit[3], self.height))
        self.Xs = np.vstack([self.X.reshape(self.Y.size), self.Y.reshape(self.Y.size)]).T

        self.first_frame = True

        #local map
        self.local_width = 80 * 5
        self.local_height = 80 * 5
        self.local_map_size = self.local_width * self.local_height
        self.local_map_limit = [-4.0, 4.0, -4.0, 4.0]
        self.local_map = .5 * np.ones((self.local_width, self.local_height), dtype=np.float)
        self.local_map_mu = .5 * np.ones((self.local_width, self.local_height), dtype=np.float)
        self.local_map_sigma = np.ones((self.local_width, self.local_height), dtype=np.float)
        self.local_X, self.local_Y = np.meshgrid(
            np.linspace(self.local_map_limit[0], self.local_map_limit[1], self.local_width),
            np.linspace(self.local_map_limit[2], self.local_map_limit[3], self.local_height))
        self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T
        #robot property
        self.current_pose = None
        self.meanfunc = pyGPs.mean.Zero()
        # self.covfunc = pyGPs.cov.Matern(d=5) * pyGPs.cov.RBFard(log_ell_list=[-0.17509193324973021, -0.3272345098877003], D=2,
        # log_sigma=-1.1789938908586013)
        self.covfunc = pyGPs.cov.Matern(d = 7)
        self.scan = None
        self.max_range = 5.66
        self.hyp_learned = False
        self.use_offline_hyp = True
        self.off_mean_hyp = [-0.73055028462998106]
        self.off_cov_hyp = [0.642442605255366, 2.095394555506737]
        self.off_lik_hyp = [-1.4860463474674306]

        #opt
        self.m = None
#self.opt = gpflow.train.AdamOptimizer(0.01)
        self.scan_skip = 10

        #time recording
        self.timeTable = np.zeros((100))
        self.times = 0

    def set_scan(self, scan):
        self.scan = scan

    def logistic_reg(self, mu, sigma, alpha=100, beta=0):
        prob = norm.cdf((alpha*mu+beta)/(1+(alpha*sigma)**2))
#amap = copy.deepcopy(np.reshape(prob, (awidth, aheight)))
#        amap_mu = copy.deepcopy(np.reshape(mu, (awidth, aheight)))
#        asigma = copy.deepcopy(np.reshape(sigma, (awidth, height)))
        return prob

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
        #outline
#self.first_link = (self.x.copy(), self.y.copy())
    def get_negative_sample(self):

        first_free_x = np.zeros((1,2))
        first_free_y = np.zeros((1,1))-1.
        
        num_of_free = 0
        first_triggered = False

        for i in range(self.scan.ranges.size):
            if self.scan.ranges[i] != np.Inf and np.mod(i, self.scan_skip) == 0:
                num_of_free += 1
                d_max = self.scan.ranges[i] - 1.5 * self.td_res
                dist = 0.0

                first_free = True
                while dist < d_max:
                        
                    dist += self.td_res
                    x1 = self.current_pose[0] + dist * np.cos(
                        self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                    x2 = self.current_pose[1] + dist * np.sin(
                        self.current_pose[2] + (self.scan.angle_min + (i + 1) * self.scan.angle_increment))
                    self.x = np.vstack([self.x, [x1, x2]])
                    self.y = np.vstack([self.y, [-1.]])

    def update_map(self):
        ix = int(np.round(self.current_pose[0] / self.map_res) + (self.width / 2) - (self.local_width / 2))
        iy = int(np.round(self.current_pose[1] / self.map_res) + (self.height / 2) - (self.local_height / 2))
        if self.first_frame:
            self.local_map_sigma *= 1000
            self.first_frame = False
        #6
        st = time.time()
        self.map_mu[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]],  self.map_sigma[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]]  = fusion_bcm(self.local_map_mu, self.map_mu[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]],
                    self.local_map_sigma, self.map_sigma[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]])
        self.timeTable[6] = ( self.timeTable[6]*(self.times-1)+( time.time()-st) ) /self.times

        #7
        st = time.time()
        self.map = self.logistic_reg(self.map_mu,self.map_sigma)
        self.timeTable[7] = ( self.timeTable[7]*(self.times-1)+( time.time()-st) ) /self.times

    def build_map(self):
        self.times += 1
        #get the true position of each range point & get the occupy observation
        #0
        st = time.time()
        self.transform2global()
        self.timeTable[0] = ( self.timeTable[0]*(self.times-1)+( time.time()-st) ) /self.times
        #1
        st = time.time()
        #get the free observation
        self.get_negative_sample()
        self.timeTable[1] = ( self.timeTable[1]*(self.times-1)+( time.time()-st) ) /self.times

        #2
        st = time.time()

        #observation

        #test point
        self.local_X, self.local_Y = np.meshgrid(
            np.linspace(self.local_map_limit[0] + self.current_pose[0], self.local_map_limit[1] + self.current_pose[0],
                        self.local_width),
            np.linspace(self.local_map_limit[2] + self.current_pose[1], self.local_map_limit[3] + self.current_pose[1],
                        self.local_height))

        self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T
        self.timeTable[2] = ( self.timeTable[2]*(self.times-1)+( time.time()-st) ) /self.times


        #3
        st = time.time()

        self.setData(self.x, self.y)
        self.timeTable[3] = ( self.timeTable[3]*(self.times-1)+( time.time()-st) ) /self.times
        #build model
        if not self.hyp_learned:
            self.optimize()
            self.hyp_learned = True
        else:
            self.meanfunc.hyp = self.off_mean_hyp
            self.covfunc.hyp = self.off_cov_hyp
            self.likfunc.hyp = self.off_lik_hyp
            #4
            st = time.time()
            self.getPosterior()
            self.timeTable[4] = ( self.timeTable[4]*(self.times-2)+( time.time()-st) ) /(self.times-1)
        #5
        st = time.time()
        self.predict(self.xs)  
        self.timeTable[5] = ( self.timeTable[5]*(self.times-1)+( time.time()-st) ) /self.times

        self.ysmu, self.yssigma = self.ym, self.ys2


        self.local_map_mu = copy.deepcopy(np.reshape(self.ysmu, (self.local_width, self.local_height)))
        self.local_map_sigma = copy.deepcopy(np.reshape(self.yssigma, (self.local_width, self.local_height)))

        self.update_map()



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


