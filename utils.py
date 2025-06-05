import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
from scipy.signal.windows import gaussian

def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z
def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth

def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def random_uniform_like(tensor, min_val, max_val):
    return (max_val - min_val) * torch.rand_like(tensor) + min_val


def sample_from_discretized_mix_logistic(y, img_channels=3, log_scale_min=-7.):
    """

    :param y: Tensor, shape=(batch_size, 3 * num_mixtures * img_channels, height, width),
    :return: Tensor: sample in range of [-1, 1]
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y.chunk(3, dim=1)

    temp = random_uniform_like(logit_probs, min_val=1e-5, max_val=1. - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))

    ones = torch.eye(means.size(1) // img_channels, dtype=means.dtype, device=means.device)

    sample = []
    for logit_prob, mean, log_scale, tmp in zip(logit_probs.chunk(img_channels, dim=1),
                                                means.chunk(img_channels, dim=1),
                                                log_scales.chunk(img_channels, dim=1),
                                                temp.chunk(img_channels, dim=1)):
        # (batch_size, height, width)
        argmax = torch.max(tmp, dim=1)[1]
        B, H, W = argmax.shape

        one_hot = ones.index_select(0, argmax.flatten())
        one_hot = one_hot.view(B, H, W, mean.size(1)).permute(0, 3, 1, 2).contiguous()

        # (batch_size, 1, height, width)
        mean = torch.sum(mean * one_hot, dim=1)
        log_scale = torch.clamp_max(torch.sum(log_scale * one_hot, dim=1), log_scale_min)

        u = random_uniform_like(mean, min_val=1e-5, max_val=1. - 1e-5)
        x = mean + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))
        sample.append(x)

    # (batch_size, img_channels, height, width)
    sample = torch.stack(sample, dim=1)

    return sample

def get_mask_imputation(length,missing_rate):
    mask=torch.zeros(length, dtype=torch.bool)

    # np.random.choice(range(length), 10, replace=np.floor((missing_rate/100)*length).astype('int'))
    true_indx=np.random.choice(range(length), np.floor((missing_rate/100)*length).astype('int'), replace=False)
    mask[true_indx]=1
    return mask

def get_mask_forcasting(length,forcasting_rate):
    mask=torch.zeros(length, dtype=torch.bool)
    true_indx=torch.arange(np.floor((100-forcasting_rate)/100*length).astype('int'), length)
    mask[true_indx]=1
    return mask

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def gaussian_kernel_smoother(y, sigma, window):
    b = gaussian(window, sigma)
    y_smooth = np.zeros(y.shape)
    neurons = y.shape[1]
    for neuron in range(neurons):
        y_smooth[:, neuron] = np.convolve(y[:, neuron], b/b.sum(),'same')
    return y_smooth
