
import numpy as np
from scipy.cluster.vq import *
import pyGPs
import torch
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

from implicitRep import PointNet, predict_2


reso = 0.05#0. as the base
expand_coeff = 3
ceoff = int(expand_coeff * 0.1 / reso)


def fusion_bcm(a, b, sa, sb):
    sz = 1. / (1. / sa + 1. / sb)
    z = sz * ((1. / sa) * a + (1. / sb) * b)
    return z, sz


def threshold(elem):
    if elem > 0.65:
        return 100
    elif elem < 0.4:
        return 0
    else:
        return -1

class GPRMap(pyGPs.GPR):
    def __init__(self, mcmc = False):
        super(GPRMap, self).__init__()
        #mode
        self.mcmc = mcmc
        #map property
        self.width = 300 * ceoff
        self.height = 300 * ceoff
        self.map_size = self.width * self.height
        self.map_limit = [x*expand_coeff for x in [-15.0, 15.0, -15.0, 15.0]]# * expand_coeff
        self.map_res = (self.map_limit[1] - self.map_limit[0]) / self.width

        self.td_res = 0.25
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
        #oneline gpom stuff
        self.first_link = None
        self.second_link = None
        self.third_link = None


        #local map
        self.local_width = 80 * ceoff
        self.local_height = 80 * ceoff
        self.local_map_size = self.local_width * self.local_height
        self.local_map_limit =[x * expand_coeff for x in [-4.0, 4.0, -4.0, 4.0] ]
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

        # Deep Learning
        self.model = PointNet(dim_k=1024)
        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('/home/yijun/Desktop/GPOM/model.pth'))


        '''
        with gpflow.defer_build():
            k = gpflow.kernels.Matern32(2) + gpflow.kernels.Bias(1)
            l = gpflow.likelihoods.StudentT()
        '''
        self.scan_skip = 3


        #threshold grid to publish
        #self.flat_grid = self.map.copy().astype(int)

        #threshold (not used)
        self.threshold = np.vectorize(threshold)

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
        self.first_link = (self.x.copy(), self.y.copy())
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
                    #second line
                    if first_free and dist >= d_max:
                        first_free = False
                        if(not first_triggered):
                            first_free_x = np.array([[x1,x2]])
                        else:
                            first_free_x = np.vstack([first_free_x, [x1,x2]])
                            first_free_y = np.vstack([first_free_y, [-1.]])
                        first_triggered = True
                #no such point:
                if first_free:
                    if(not first_triggered):
                            first_free_x = np.array([[self.current_pose[0],self.current_pose[1]]])
                    else:
                        first_free_x = np.vstack([first_free_x,[self.current_pose[0],self.current_pose[1]]])
                        first_free_y = np.vstack([first_free_y, [-1.]])
        self.second_link = (first_free_x, first_free_y)


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
        '''
        for i in range(self.local_map_mu.shape[0]):
            for j in range(self.local_map_mu.shape[1]):
                if not (ix < 0 and iy < 0):
                    if ix < self.width and iy < self.height:
                        if self.first_frame:
                            self.local_map_sigma[i, j] *= 1000
                            self.first_frame = False

                        z, sz = fusion_bcm(self.local_map_mu[i, j], self.map_mu[i + ix, j + iy], self.local_map_sigma[i, j],
                                           self.map_sigma[i + ix, j + iy])
                        self.map_mu[i + ix, j + iy] = copy.deepcopy(z)
                        self.map_sigma[i + ix, j + iy] = copy.deepcopy(sz)
        '''
        self.timeTable[6] = ( self.timeTable[6]*(self.times-1)+( time.time()-st) ) /self.times

        #7
        st = time.time()

        self.map[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]] = self.logistic_reg(
            self.map_mu[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]],
            self.map_sigma[iy:iy+self.local_map_mu.shape[0] , ix:ix+self.local_map_mu.shape[1]]
        )
        #self.map = self.logistic_reg(self.map_mu,self.map_sigma)
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

        '''
        with open('/home/yijun/Desktop/GPOM/test.npy','wb') as f:
            np.save(f,self.x)
            np.save(f,self.y)
        print('saved')
        '''

        #3
        st = time.time()

        #observation

        '''
        if not self.first_frame:
            self.m.X = np.vstack([self.first_link[0], self.second_link[0]])
            self.m.Y = np.vstack([self.first_link[1], self.second_link[1]])
            print(self.m.X.shape)
        '''
        ax,ay=np.vstack([self.first_link[0], self.second_link[0]]), np.vstack([self.first_link[1], self.second_link[1]])
        self.setData(ax,ay)
        self.timeTable[3] = ( self.timeTable[3]*(self.times-1)+( time.time()-st) ) /self.times
        #2
        st = time.time()
        #test point
        '''
        self.local_X, self.local_Y = np.meshgrid(
            np.linspace(self.local_map_limit[0] + self.current_pose[0], self.local_map_limit[1] + self.current_pose[0],
                        self.local_width),
            np.linspace(self.local_map_limit[2] + self.current_pose[1], self.local_map_limit[3] + self.current_pose[1],
                        self.local_height))

        self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T
        '''
        upper_left = (self.local_map_limit[0] + self.current_pose[0],self.local_map_limit[2] + self.current_pose[1])
        dx = 2*self.local_map_limit[1] / self.local_width
        dy = 2*self.local_map_limit[3] / self.local_height
        plt_1link = self.first_link[0].copy()
        plt_2link = self.second_link[0].copy()
        plt_1link[:,0] -= upper_left[0] 
        plt_1link[:,1] -= upper_left[1] 
        plt_2link[:,0] -= upper_left[0] 
        plt_2link[:,1] -= upper_left[1] 
        
        plt_1link[:,0] /= dx 
        plt_1link[:,1] /= dy 
        plt_2link[:,0] /= dx 
        plt_2link[:,1] /= dy 
        
        first_link_mask = np.zeros((self.local_width, self.local_height))
        second_link_mask = np.zeros((self.local_width, self.local_height))


#first_link_mask = graph_in_poly(first_link_mask, plt_1link)
        occ_y, occ_x = polygon(plt_1link[:,1],plt_1link[:,0],(self.local_width, self.local_height))
#second_link_mask = graph_in_poly(second_link_mask, plt_2link)
        free_y, free_x = polygon(plt_2link[:,1],plt_2link[:,0],(self.local_width, self.local_height))
        first_link_mask[occ_y,occ_x] = 1
        second_link_mask[free_y,free_x] = 1




        ring_mask = first_link_mask - second_link_mask 
        ring_y, ring_x = np.where(ring_mask>=1)
#free_y, free_x = np.where(second_link_mask>=1)
        ring_ybc = ring_y*dy
        ring_xbc = ring_x*dx
        ring_ybc += upper_left[1]
        ring_xbc += upper_left[0]
        self.xs = np.vstack([ring_xbc.reshape(-1), ring_ybc.reshape(-1)]).T
        self.timeTable[2] = ( self.timeTable[2]*(self.times-1)+( time.time()-st) ) /self.times


        #build model
        center = np.array(self.current_pose).reshape((1,3))[:,:2]
        x = self.x-center
        x_max = np.absolute(x).max(axis=0,keepdims=True)
        x /= (x_max*4) # to be in [-.25,.25]

        x_ = self.xs - center
        x_ /= (x_max*4)


        X_train = torch.from_numpy(x).type(torch.float).to(self.device)
        y_train = torch.from_numpy(self.y).type(torch.float).to(self.device)
        X_test = torch.from_numpy(x_).type(torch.float).to(self.device)

        pred_mu, pred_sigma = predict_2(X_test.unsqueeze(0), X_train.unsqueeze(0), y_train.unsqueeze(0), self.model,s_p_2=1,sigma_2=1e-2)





        self.ysmu = np.zeros((self.local_width, self.local_height))
        self.yssigma = np.ones((self.local_width, self.local_height)) * 1000
        '''
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
        '''
        #5
        st = time.time()
        pred_mu, pred_sigma = predict_2(X_test.unsqueeze(0), X_train.unsqueeze(0), y_train.unsqueeze(0), self.model,s_p_2=1,sigma_2=1e-2)
        #self.predict(self.xs)  
        self.timeTable[5] = ( self.timeTable[5]*(self.times-1)+( time.time()-st) ) /self.times

        #pred_mu, pred_sigma= self.ym, self.ys2
        pred_mu = pred_mu.cpu().detach().numpy()
        pred_sigma = pred_sigma.cpu().detach().numpy()

        self.ysmu[(ring_y,ring_x)], self.yssigma[(ring_y,ring_x)] = pred_mu.reshape(-1), pred_sigma.reshape(-1)


        self.ysmu[free_y, free_x] = -1.
        self.yssigma[free_y, free_x] = 0.1

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


