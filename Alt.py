import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import Dataset

import math
from torch.autograd import Variable
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
        # self.enc2 = encoder_block(input_dim * 2 + d_model, input_dim)
        # self.enc3 = encoder_block(input_dim * 2 + d_model, input_dim)

        # self.dec1 = encoder_block(input_dim * 2 + d_model, input_dim)
        # self.dec2 = encoder_block(input_dim * 2 + d_model, input_dim)
        self.dec3 = encoder_block(input_dim * 2 + d_model, output_dim)


    def forward(self, x,ts):
        if x.ndim == 2:
            x = x.unsqueeze(0)
            embedded_k = get_timestep_embedding(ts, self.d_model)
            embedded_k=embedded_k.repeat(x.shape[1], 1)
            embedded_k = embedded_k.unsqueeze(0)
            x_in = self.enc1(torch.cat((x, embedded_k), -1))
            # x_in = self.enc2(torch.cat((x,x_in, embedded_k), -1))
            # x_in = self.enc3(torch.cat((x,x_in, embedded_k), -1))
            # x_in = self.dec1(torch.cat((x,x_in, embedded_k), -1))
            # x_in = self.dec2(torch.cat((x,x_in, embedded_k), -1))
            x_in= self.dec3(torch.cat((x,x_in, embedded_k), -1))
            if self.attention:
                for attn_layer in self.attention_layers:
                    x_in = attn_layer(x_in)

            output=x_in
        else:
            output=0
        return output



def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps=torch.tensor([timesteps])
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


class Alt(nn.Module):
    def __init__(self, latent_dim, obser_dim, sigma_x, alpha,importance_sample_size,n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.obser_dim =torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.sigma_x=torch.tensor([sigma_x], requires_grad=False).to(self.device)

        self.alpha = torch.tensor([alpha], requires_grad=False).to(self.device)
        self.sigma_z = self.alpha* self.sigma_x
        self.importance_sample_size= importance_sample_size
        self.n_layers=n_layers
        self.dp_rate=.0
        # self.f_phi = SelfAttentionTimeSeries(input_dim=self.latent_dim,
        #                         output_dim=self.latent_dim,
        #                         d_model=10,
        #                         num_heads=1,attention=False,
        #                         num_layers=self.n_layers,device=self.device)

        self.g_theta =SelfAttentionTimeSeries(input_dim=self.latent_dim,
                                output_dim=self.obser_dim,
                                d_model=10,
                                num_heads=1,attention=False,
                                num_layers=self.n_layers,device=self.device)
        self.f_phi_x = SelfAttentionTimeSeries(input_dim=self.obser_dim,
                                               output_dim=self.latent_dim,
                                               d_model=10,
                                               num_heads=1, attention=False,
                                               num_layers=self.n_layers, device=self.device)

        # self.alpha_encoder = nn.Linear(1, 1)


    def forward(self, obsrv,mask, eps_x,eps_z,obsr_enable,smoother_enable):
        # obsrv = [src len, 1, obsr dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        self.obsrv = obsrv#.repeat(1,self.importance_sample_size,1)
        batch_size = self.obsrv.shape[1]
        seq_len = self.obsrv.shape[0]

        # tensor to store decoder outputs
        self.z_hat = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        self.x_hat = torch.zeros(seq_len, batch_size, self.obser_dim).to(self.device)
        self.z_x_hat=torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)

        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder


        self.z_hat[0] =  self.sigma_x*Variable(torch.randn(self.z_x_hat[0].shape)).to(self.device)

        for k in range(1, seq_len):

            eps_z = (eps_z+ self.sigma_z *torch.randn(self.z_hat[0].shape).to(self.device))/2
            eps_x = ( eps_x+ self.sigma_x *torch.randn(self.x_hat[0].shape).to(self.device))/2


            if obsr_enable:
                if mask[k]:
                    self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                        self.z_hat[k - 1:k].clone(),
                        k) + eps_x

                    self.z_hat[k] = torch.sqrt(self.alpha) * self.f_phi_x(
                        self.x_hat[k ], k) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * \
                                      self.z_hat[
                                          k - 1].clone() + eps_z

                else:

                    self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) *self.g_theta(self.z_hat[k - 1:k].clone(),k) +eps_x

                    self.z_hat[k] =torch.sqrt(self.alpha) * self.f_phi_x(
                            self.obsrv[k],k)+ torch.sqrt(1-self.alpha- torch.pow(self.sigma_z,2)) *self.z_hat[k-1].clone() + eps_z

            else:

                self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(self.z_hat[k - 1:k].clone(),
                                                                                          k) + eps_x
                # self.x_hat[k] = torch.sqrt(1-torch.pow(self.sigma_x,2))* self.g_theta(self.z_hat[k - 1].clone())+eps_x
                self.z_hat[k] = torch.sqrt(self.alpha) * self.f_phi_x(
                    self.x_hat[k], k) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * self.z_hat[
                                      k - 1].clone() + eps_z

        if smoother_enable:
            for k in range(1,seq_len)[::-1]:

                eps_z = (eps_z + self.sigma_z * torch.randn(self.z_hat[0].shape).to(self.device)) / 2
                eps_x = (eps_x + self.sigma_x * torch.randn(self.x_hat[0].shape).to(self.device)) / 2

                if obsr_enable:
                    if mask[k]:
                        self.x_hat[k-1] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                            self.z_hat[k].clone(),
                            k-1) + eps_x

                        self.z_hat[k-1] = torch.sqrt(self.alpha) * self.f_phi_x(
                            self.x_hat[k-1], k-1) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * \
                                        self.z_hat[
                                            k ].clone() + eps_z

                    else:

                        self.x_hat[k-1] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                            self.z_hat[k].clone(), k-1) + eps_x

                        self.z_hat[k-1] = torch.sqrt(self.alpha) * self.f_phi_x(
                            self.obsrv[k-1], k -1) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * \
                                        self.z_hat[k ].clone() + eps_z

                else:

                    self.x_hat[k-1] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                        self.z_hat[k].clone(),
                        k-1) + eps_x
                    # self.x_hat[k] = torch.sqrt(1-torch.pow(self.sigma_x,2))* self.g_theta(self.z_hat[k - 1].clone())+eps_x
                    self.z_hat[k-1] = torch.sqrt(self.alpha) * self.f_phi_x(
                        self.x_hat[k-1], k-1) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * self.z_hat[
                                        k ].clone() + eps_z

            for k in range(1, seq_len):

                eps_z = (eps_z + self.sigma_z * torch.randn(self.z_hat[0].shape).to(self.device)) / 2
                eps_x = (eps_x + self.sigma_x * torch.randn(self.x_hat[0].shape).to(self.device)) / 2

                if obsr_enable:
                    if mask[k]:
                        self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                            self.z_hat[k - 1:k].clone(),
                            k) + eps_x

                        self.z_hat[k] = torch.sqrt(self.alpha) * self.f_phi_x(
                            self.x_hat[k], k) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * \
                                        self.z_hat[
                                            k - 1].clone() + eps_z

                    else:

                        self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                            self.z_hat[k - 1:k].clone(), k) + eps_x

                        self.z_hat[k] = torch.sqrt(self.alpha) * self.f_phi_x(
                            self.obsrv[k], k) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * self.z_hat[
                                            k - 1].clone() + eps_z

                else:

                    self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.g_theta(
                        self.z_hat[k - 1:k].clone(),
                        k) + eps_x
                    # self.x_hat[k] = torch.sqrt(1-torch.pow(self.sigma_x,2))* self.g_theta(self.z_hat[k - 1].clone())+eps_x
                    self.z_hat[k] = torch.sqrt(self.alpha) * self.f_phi_x(
                        self.x_hat[k], k) + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * self.z_hat[
                                        k - 1].clone() + eps_z
        self.z_hat[1:]=self.z_hat[1:].detach()
        return self.z_hat


    def loss (self, a,b,c,z):
        L1 = F.mse_loss(torch.sqrt(1 - torch.pow(self.sigma_x, 2)) * self.x_hat, self.obsrv)*self.sigma_z**2


        L2 = F.mse_loss( z[1:], torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z, 2)) * self.z_hat[:-1]+
                        torch.sqrt(self.alpha) * self.z_x_hat[1:])*self.sigma_x**2

        L = b * L2 + a * L1
        print('L1=%f, L2=%f, L=%f' % (L1, L2, L))
        return L



class get_dataset(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, x,z, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.z[index]]

class get_dataset_HC(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, x,z,v, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, index):
        return [self.x[:,index,:], self.z[:,index,:],self.v[:,index,:]]

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.05, 0.05)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
