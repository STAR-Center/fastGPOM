
import numpy as np
from scipy.cluster.vq import *
#import pyGPs
import gpflow
import copy
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PoseArray
from scipy.stats import norm
import random
from mygp import mygp
from thrd import MyThread
import pdb

def bst_predict( bst_data, bst_label, test_data_in,  i):
    if len(test_data_in) == 0 or bst_data.shape[0] == 0:
        return (None, None, i)
#bst_models.X = bst_data.astype(np.float64)#gpflow.params.DataHolder(bst_data)
#bst_models.Y = (bst_label.reshape((-1,1))*2-1.).astype(np.float64)#gpflow.params.DataHolder(bst_label)
    local_mu_lst, local_sigma_lst = mygp(bst_data, bst_label*2-1., test_data_in)
    return (local_mu_lst.reshape((-1,1)), local_sigma_lst.reshape((-1,1)), i)



def fusion_bcm(a, b, sa, sb):
    sz = 1. / (1. / sa + 1. / sb)
    z = sz * ((1. / sa) * a + (1. / sb) * b)
    return z, sz

class GPRMap:
    def __init__(self, mcmc = False):
        self.boost_nm = 5
        #mode
        self.mcmc = mcmc
        #map property
        self.width = 300
        self.height = 300
        self.map_size = self.width * self.height
        self.map_limit = [-15.0, 15.0, -15.0, 15.0]
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
        self.local_width = 80
        self.local_height = 80
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

        #opt
        self.m = None
        self.opt = gpflow.train.AdamOptimizer(0.01)
        '''
        with gpflow.defer_build():
            k = gpflow.kernels.Matern32(2) + gpflow.kernels.Bias(1)
            l = gpflow.likelihoods.StudentT()
        '''
        self.scan_skip = 10

    def set_scan(self, scan):
        self.scan = scan

    def logistic_reg(self, mu, sigma, alpha=1, beta=1):
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

    def get_negative_sample(self):
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




    def update_map(self):
        iy = int(np.round(self.current_pose[0] / self.map_res) + (self.width / 2) - (self.local_width / 2))
        ix = int(np.round(self.current_pose[1] / self.map_res) + (self.height / 2) - (self.local_height / 2))
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
        self.map = self.logistic_reg(self.map_mu,self.map_sigma)

    def data_boost(self,data, label):
        tot_nm = label.shape[0]
        shuffled_id = list(range(tot_nm))
        random.shuffle(shuffled_id)
        shuffled_id = np.array(shuffled_id)
        st_id = 0
        step = int(tot_nm/self.boost_nm)
        ret_data = [None]*self.boost_nm 
        ret_label = [None]*self.boost_nm 
        for i in range(self.boost_nm):
            ret_data[i] = data[(shuffled_id[st_id:min(st_id+step,tot_nm)]),:]
            ret_label[i] = label[(shuffled_id[st_id:min(st_id+step,tot_nm)])]
            st_id = min(st_id+step, tot_nm)
        return ret_data, ret_label

 
    def build_map(self):
        #get the true position of each range point & get the occupy observation
        self.transform2global()
        #get the free observation
        self.get_negative_sample()
        #test point
        self.local_X, self.local_Y = np.meshgrid(
            np.linspace(self.local_map_limit[0] + self.current_pose[0], self.local_map_limit[1] + self.current_pose[0],
                        self.local_width),
            np.linspace(self.local_map_limit[2] + self.current_pose[1], self.local_map_limit[3] + self.current_pose[1],
                        self.local_height))

        self.xs = np.vstack([self.local_X.reshape(self.local_Y.size), self.local_Y.reshape(self.local_Y.size)]).T

        #build model
        if self.mcmc:#mcmc
            if self.first_frame:
                with gpflow.defer_build():
                    k = gpflow.kernels.Matern32(2) + gpflow.kernels.Bias(1)
                    l = gpflow.likelihoods.StudentT()
                    self.m = gpflow.models.GPMC(self.x, self.y, k, l)
                    #kernel
                    self.m.kern.matern32.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
                    self.m.kern.matern32.variance.prior = gpflow.priors.Gamma(1.,1.)
                    self.m.kern.bias.variance.prior = gpflow.priors.Gamma(1.,1.)

                    #estimate the hyperparam
                    self.m.compile()
                    self.opt.minimize(self.m,maxiter=15)#MAP

                    s = gpflow.train.HMC()
                    self.samples = s.sample(self.m, 20, epsilon = 0.2, lmax = 20, lmin = 5, thin = 5, logprobs = False)

                self.first_frame = False

            f_samples = []
            for i, s in self.samples.iterrows():
                self.m.assign(s)
                f_samples.append(self.m.predict_f_samples(self.xs, 5, initialize = False))
            f_samples = np.vstack(f_samples)
            self.ysmu = np.mean(f_samples, 0)
            self.yssigma = np.var(f_samples, 0)
                

        else:#no mcmc
            '''
            if self.first_frame:
                #estimate the hyperparam
                with gpflow.defer_build():
                    #model
                    k = gpflow.kernels.Matern52(2)
                    print (self.x[:,:].shape, self.y[:,:].shape)
                    self.m = gpflow.models.GPR(self.x, self.y, k)
                    self.m.likelihood.variance = 0.01
                    #kernel
                    

                # MAP or say MLE because the prior is uniform
                self.m.compile() 
#gpflow.training.scipy_optimizer().minimize(self.m)
                self.opt = gpflow.train.ScipyOptimizer()
                self.opt.minimize(self.m)

                self.fitst_frame = False

            self.ysmu, self.yssigma = self.m.predict_y(self.xs)
            '''        
            bst_data, bst_label= self.data_boost(self.x,self.y)
            local_mu_lst, local_sigma_lst = [None]*self.boost_nm, [None]*self.boost_nm
            results = []
            li = []
            for bst_i in range(self.boost_nm):
                t = MyThread(bst_predict, (bst_data[bst_i], bst_label[bst_i], self.xs,  bst_i))
                li.append(t)
                t.start()
            for t in li:
                t.join()
            for t in li:
                results.append(t.get_result())
            for res in results:
                '''
                if res is None:
                    continue
                if res[0] is None:
                    continue
                '''
                bst_mu, bst_sigma , bst_i = res
                local_mu_lst[bst_i], local_sigma_lst[bst_i] = bst_mu, bst_sigma
            if local_mu_lst[0] is not None:
                for bst_i in range(1,self.boost_nm):
                    local_mu_lst[0], local_sigma_lst[0] = fusion_bcm(local_mu_lst[0], local_mu_lst[bst_i], local_sigma_lst[0], local_sigma_lst[bst_i])

            #get the mu and sigma
            self.ysmu, self.yssigma = local_mu_lst[0], local_sigma_lst[0]

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


