import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch.distributions import Categorical, MultivariateNormal, Normal
class lorenz_sample_generator():
    def __init__(self,config):

        self.s=config['s']
        self.r=config['r']
        self.b=config['b']
        self.dt= config['dt']
        self.num_steps=config['traj_size']
        self.initial_val =config['initial_val']
        self.noise_std = config['noise_std']
        self.noise_enable = config['noise_enable']
        self.scale_range = config['scale_range']
    def lorenz_one_sample(self,x, y, z):
        """
        Given:
           x, y, z: a point of interest in three dimensional space
           s, r, b: parameters defining the lorenz attractor
        Returns:
           x_dot, y_dot, z_dot: values of the lorenz attractor's partial
               derivatives at the point x, y, z
        """
        x_dot = self.s*(y - x)
        y_dot = self.r*x - y - x*z
        z_dot = x*y - self.b*z
        return x_dot, y_dot, z_dot
    def sample_trajectory(self):
        # Need one more for the initial values
        xs = np.empty(self.num_steps + 1)
        ys = np.empty(self.num_steps + 1)
        zs = np.empty(self.num_steps + 1)
        MVN=MultivariateNormal(torch.tensor(torch.zeros(3,)),.1*torch.tensor(torch.diag(torch.ones(3,))))
        # Set initial values
        xs[0], ys[0], zs[0] = self.initial_val
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(self.num_steps):

            if self.noise_enable:
                x_dot, y_dot, z_dot = self.lorenz_one_sample(xs[i], ys[i], zs[i])
                noise_sampel = MVN.sample((1,))
                noise_sampel = noise_sampel.detach().numpy().squeeze()
                xs[i + 1] = xs[i]+(noise_sampel[0]+noise_sampel[1])/2 + (x_dot * self.dt)
                ys[i + 1] = ys[i] +(noise_sampel[0]+noise_sampel[2])/2+ (y_dot * self.dt)
                zs[i + 1] = zs[i] +(noise_sampel[0]+ noise_sampel[1]+noise_sampel[2])/3+ (z_dot * self.dt)
            else:

                x_dot, y_dot, z_dot = self.lorenz_one_sample(xs[i], ys[i], zs[i])
                xs[i + 1] = xs[i]  + (x_dot * self.dt)
                ys[i + 1] = ys[i]  + (y_dot * self.dt)
                zs[i + 1] = zs[i] + (z_dot * self.dt)

        # normalize the value of the xs, ys, zs
        min_max_scaler = preprocessing.MinMaxScaler()
        xs= (min_max_scaler.fit_transform(xs.reshape([-1,1]))-0.5)*(self.scale_range[1]-self.scale_range[0])
        ys = (min_max_scaler.fit_transform(ys.reshape([-1,1])) - 0.5) * (self.scale_range[1] - self.scale_range[0])
        zs = (min_max_scaler.fit_transform(zs.reshape([-1,1])) - 0.5) * (self.scale_range[1] - self.scale_range[0])
        return xs,ys,zs

    def generate_spikes(self,xs, ys, zs, number_of_observations, min_sigma,max_hist_dependency_Langevin_data,max_firing_rate):

        max_sigma = (self.scale_range[1] - self.scale_range[0]) / 20

        bfr = np.random.uniform(0, high=max_firing_rate, size=number_of_observations)

        mu_x = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1],
                                 size=number_of_observations)
        mu_y = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1],
                                 size=number_of_observations)
        mu_z = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1],
                                 size=number_of_observations)

        sigma_x = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)
        sigma_y = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)
        sigma_z = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)

        Lambdas1 = np.exp(
            bfr - (xs - mu_x) ** 2 / (2 * sigma_x ** 2) - (ys - mu_y) ** 2 / (2 * sigma_y ** 2) - (zs - mu_z) ** 2 / (
                        2 * sigma_z ** 2))
        Spikes = (np.random.poisson(Lambdas1))
        Spikes[Spikes > 1] = 1

        Lambdas2 = np.zeros_like(Lambdas1)

        sigma_eff = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)
        rt = np.random.uniform(0, 0, size=number_of_observations)
        for ii in range(number_of_observations):
            qq=np.random.randint(0,number_of_observations)
            si=np.where(Spikes[:,qq] >0)[0]
            if len(si)>0:
                Lambdas2[:,ii] = np.nanmean(1-np.exp( -
                                    (np.arange(Spikes.shape[0]).reshape([-1,1]) - si.reshape([1,-1]) -rt[ii]) ** 2 / (2 * sigma_eff[ii] ** 2) ),axis=-1).squeeze()

        Spikes = (np.random.poisson(Lambdas1*Lambdas2))
        Spikes[Spikes > 1] = 1
        return Spikes









