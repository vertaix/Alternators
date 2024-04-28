
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import einops
import math
from torch.autograd import Variable
from sklearn.neighbors import KernelDensity
class encoder_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(encoder_block, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim, bias=True),  # First linear layer
            nn.GELU(),  # ReLU activation
            nn.Linear(in_dim, in_dim, bias=True),  # First linear layer
            nn.GELU(),  # ReLU activation
            nn.Linear(in_dim, out_dim, bias=True))  # Second linear layer

    def forward(self, x):
        return  self.net(x)

class Swish(nn.Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(.1)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x = x.permute(1, 0, 2)  # Reshape for MultiheadAttention
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output=self.norm(x + self.dropout(attn_output))
        # attn_output = attn_output.permute(1, 0, 2)
        return attn_output

class SelfAttentionTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads,attention, num_layers,device):
        super(SelfAttentionTimeSeries, self).__init__()
        # self.max_seq_len=max_seq_len
        self.device=device
        self.d_model=d_model
        self.attention=attention
        # self.pos_encoding = PositionalEncoding(d_model, 0,max_seq_len)
        self.attention_layers = nn.ModuleList([SelfAttention(output_dim, num_heads) for _ in range(num_layers)])

        self.enc1 =  encoder_block(input_dim + d_model, input_dim)
        self.enc2 = encoder_block(input_dim * 2 + d_model, input_dim)
        self.enc3 = encoder_block(input_dim * 2 + d_model, input_dim)

        self.dec1 = encoder_block(input_dim * 2 + d_model, input_dim)
        self.dec2 = encoder_block(input_dim * 2 + d_model, input_dim)
        self.dec3 = encoder_block(input_dim * 2 + d_model, output_dim)


    def forward(self, x,ts):
        if x.ndim == 2:
            x = x.unsqueeze(0)
            embedded_k = get_timestep_embedding(ts, self.d_model,self.device)
            embedded_k=embedded_k.repeat(x.shape[1], 1)
            embedded_k = embedded_k.unsqueeze(0)
            x_in = self.enc1(torch.cat((x, embedded_k), -1))
            x_in = self.enc2(torch.cat((x,x_in, embedded_k), -1))
            x_in = self.enc3(torch.cat((x,x_in, embedded_k), -1))
            x_in = self.dec1(torch.cat((x,x_in, embedded_k), -1))
            x_in = self.dec2(torch.cat((x,x_in, embedded_k), -1))
            x_in= self.dec3(torch.cat((x,x_in, embedded_k), -1))
            if self.attention:
                for attn_layer in self.attention_layers:
                    x_in = attn_layer(x_in)

            output=x_in
        else:
            output=0
        return output
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.25
    return torch.linspace(beta_start, beta_end, timesteps)

def get_timestep_embedding(timesteps, embedding_dim,device):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps=torch.tensor([timesteps]).to(device)
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
class Alt_static(nn.Module):
    def __init__(self, latent_dim, obser_dim,noise_max,n_layers, number_step, alpha, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.number_step=number_step
        self.sigma_x=torch.linspace( noise_max , 0.01,steps=number_step, requires_grad=False).to(self.device)
        self.alpha_k=.5+0*torch.linspace( 0.01,1-np.max([noise_max, alpha*noise_max]) ,steps=number_step, requires_grad=False).to(self.device)
        self.alpha = torch.tensor([alpha], requires_grad=False).to(self.device)
        self.sigma_z = self.sigma_x#/ self.alpha
        
        self.n_layers=n_layers

        self.dp_rate=.1
        self.obser_dim = torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.g_theta = SelfAttentionTimeSeries(input_dim=self.latent_dim,
                                               output_dim=self.obser_dim,
                                               d_model=10,
                                               num_heads=2,
                                               attention=False,
                                               num_layers=self.n_layers, device=self.device)
        self.f_phi_x = SelfAttentionTimeSeries(input_dim=self.obser_dim,
                                               output_dim=self.latent_dim,
                                               d_model=10,
                                               num_heads=2,
                                               attention=False,
                                               num_layers=self.n_layers, device=self.device)

    def forward(self, obsrv, eps_x_org, eps_z_org, obsr_enable, enabled_noise=True):

            self.obsrv = obsrv.to(self.device)  # .repeat(1,self.importance_sample_size,1)
            batch_size = self.obsrv.shape[0]

            # tensor to store decoder outputs
            self.z_hat = torch.zeros(self.number_step, batch_size, self.latent_dim).to(self.device)
            self.x_hat = torch.zeros(self.number_step, batch_size, self.obser_dim).to(self.device)
            self.x_inf = torch.zeros(self.number_step, batch_size, self.obser_dim,requires_grad=False).to(self.device)
            self.z_x_hat = torch.zeros(self.number_step, batch_size, self.latent_dim).to(self.device)



            self.z_hat[0] =  eps_z_org

            for k in range(1, self.number_step):
                # self.x_inf[k] =   torch.sqrt(1 - torch.pow(self.sigma_x[k], 2))*self.obsrv + self.sigma_x[k] *eps_x_org
                self.x_inf[k] =self.obsrv

                eps_z = (self.sigma_z[k] * eps_z_org
                         + self.sigma_z[k] * torch.randn(self.z_hat[0].shape).to(
                    self.device)) / 2
                eps_x = (self.sigma_x[k] * eps_x_org
                         + self.sigma_x[k] * torch.randn(self.x_hat[0].shape).to(
                    self.device)) / 2

                if obsr_enable:

                    self.x_hat[k] = self.g_theta(
                        torch.sqrt(
                            1 - torch.pow(self.sigma_z[k], 2)) *
                        self.z_hat[
                            k - 1] + eps_z,
                        k)

                    self.z_x_hat[k] = self.f_phi_x(
                        torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.obsrv + eps_x,
                        k)
                    self.z_hat[k] = torch.sqrt(self.alpha) * self.z_x_hat[k] + torch.sqrt(1 - self.alpha ) * self.z_hat[
                                        k - 1]

                else:

                    if enabled_noise:

                        self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.g_theta(
                            self.z_hat[k - 1],
                            k) + eps_x

                        self.z_hat[k] = torch.sqrt(self.alpha) * self.f_phi_x(
                            self.x_hat[k],
                            k) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z[k], 2)) * self.z_hat[k - 1] + eps_z
                    else:
                        self.x_hat[k] = self.g_theta(
                            torch.sqrt(
                                1 - torch.pow(self.sigma_z[k], 2)) *
                            self.z_hat[
                                k - 1] + eps_z,
                            k)

                        self.z_x_hat[k] = self.f_phi_x(
                            torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.x_hat[k] + eps_x,
                            k)
                        self.z_hat[k] = torch.sqrt(self.alpha) * self.z_x_hat[k] + torch.sqrt(1 - self.alpha) * \
                                        self.z_hat[
                                            k - 1]
            return self.z_hat,self.x_hat

    def loss(self, a, b):
        # L1 = F.mse_loss(self.x_hat/self.sigma_x.reshape([-1,1,1]), self.x_inf/self.sigma_x.reshape([-1,1,1]))
        # L2 = F.mse_loss((self.z_hat[2:] - torch.sqrt(1 - self.alpha ) * self.z_hat[1:-1])/self.sigma_z[2:].reshape([-1,1,1]),
        #                 torch.sqrt(self.alpha) * self.z_x_hat[
        #                                          2:]/self.sigma_z[2:].reshape([-1,1,1]))
        # L1 = torch.mean(
        #     torch.mean(F.mse_loss(self.x_hat[-1], self.x_inf[-1], reduction='none')) / self.sigma_x[
        #                                                                                            -1].reshape(
        #         [-1, 1]) ** 2)
        L1 = torch.mean(
            torch.mean(F.mse_loss(self.x_hat[1:], self.x_inf[1:], reduction='none'), dim=[1, 2]) / self.sigma_x[
                                                                                                   1:].reshape(
                [-1, 1]) ** 2)
        L2 = torch.mean(torch.mean(F.mse_loss(
            (self.z_hat[1:] - torch.sqrt(1 - self.alpha_k[1:]).reshape([-1, 1, 1]) * self.z_hat[:-1]),
            torch.sqrt(self.alpha_k[1:]).reshape([-1, 1, 1]) * self.z_x_hat[
                                                               1:], reduction='none'), dim=[1, 2]) / self.sigma_z[
                                                                                                     1:].reshape(
            [-1, 1]) ** 2)

        L = b * L2 + a * L1

        return L
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plt_flow_samples(prior_sample, ax, npts=100, memory=100, kde_enable=True, title="", device="cpu"):
    z = prior_sample.to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory ** 2)):
        zk.append(z[ii])
    zk = torch.cat(zk, 0).cpu().numpy()
    x_min, x_max = prior_sample[:, 0].min(), prior_sample[:, 0].max()
    y_min, y_max = prior_sample[:, 1].min(), prior_sample[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, npts),
                         np.linspace(y_min, y_max, npts))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)

    if kde_enable:
        # Fit a kernel density estimator to the data
        kde = KernelDensity(bandwidth=0.05, kernel='tophat')
        kde.fit(zk)

        # Compute the log density values for the grid points
        log_density = kde.score_samples(grid_points)

        # Reshape the log density values to match the grid shape
        density = np.exp(log_density)
        density = density.reshape(xx.shape)
        ax.imshow(density.T, cmap='copper',  # ,extent=(-2, 3, -2, 3),
                  # interpolation='nearest',
                  origin='lower')
    else:
        # hist, x_edges, y_edges = np.histogram2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=(npts, npts))
        # ax.imshow(hist, cmap='copper',
        #           interpolation='nearest',
        #           origin='lower')
        # copper_color = (0.5, 0.3, 0.1)
        burnt_orange = "#cc5500"
        ax.scatter(zk[:, 0], zk[:, 1], c=burnt_orange, s=.01, alpha=.1)
        ax.axis('off')
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

