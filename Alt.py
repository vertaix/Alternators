import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
from torch.autograd import Variable

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (T, B, D)
        Returns:
            Tensor of shape (T, B, D)
        """
        return x + self.pe[:x.size(0)]


class SelfAttentionTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, attention, num_layers, device):
        super().__init__()
        self.device = device
        self.attention = attention
        self.d_model = d_model

        # Project input to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape (T, B, input_dim)
            t: current timestep (unused, kept for compatibility)
        Returns:
            Tensor of shape (T, B, output_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Convert to (1, B, input_dim)

        x = self.input_proj(x)           # (T, B, d_model)
        x = self.pos_encoder(x)          # Add sinusoidal time embedding
        x = self.transformer_encoder(x)  # (T, B, d_model)
        x = self.output_proj(x)          # (T, B, output_dim)
        return x

# Alternator model
class Alt(nn.Module):
    def __init__(self, latent_dim, obser_dim, sigma_x, alpha, importance_sample_size, n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.obser_dim = obser_dim
        self.sigma_x = torch.tensor(sigma_x)
        self.alpha = torch.tensor(alpha)
        self.sigma_z = alpha * sigma_x
        self.importance_sample_size = importance_sample_size
        self.n_layers = n_layers

        self.g_theta = SelfAttentionTimeSeries(
            input_dim=self.latent_dim,
            output_dim=self.obser_dim,
            d_model=64,
            num_heads=4,
            attention=True,
            num_layers=self.n_layers,
            device=self.device
        )

        self.f_phi_x = SelfAttentionTimeSeries(
            input_dim=self.obser_dim,
            output_dim=self.latent_dim,
            d_model=64,
            num_heads=4,
            attention=True,
            num_layers=self.n_layers,
            device=self.device
        )

    def forward(self, obsrv, mask, eps_x, eps_z, obsr_enable, smoother_enable=False):
        T, B, _ = obsrv.shape
        self.obsrv = obsrv
        self.z_hat = torch.zeros(T, B, self.latent_dim).to(self.device)
        self.x_hat = torch.zeros(T, B, self.obser_dim).to(self.device)

        self.z_hat[0] = self.sigma_x * torch.randn(B, self.latent_dim).to(self.device)

        for t in range(1, T):
            eps_z=self.sigma_z*torch.randn(self.z_hat[0].shape).to(self.device)
            eps_x=self.sigma_z*torch.randn(self.x_hat[0].shape).to(self.device)
            if obsr_enable and mask[t].any():
                self.x_hat[t] = torch.sqrt(1 - self.sigma_x ** 2) * self.g_theta(self.z_hat[t-1:t].clone(), t)
                self.z_hat[t] = (
                    torch.sqrt(self.alpha) * self.f_phi_x(self.obsrv[t:t+1], t)
                    + torch.sqrt(1 - self.alpha - self.sigma_z ** 2) * self.z_hat[t-1:t].clone()
                )
            else:
                self.x_hat[t] = (
                    torch.sqrt(1 - self.sigma_x ** 2) * self.g_theta(self.z_hat[t-1:t].clone(), t) + eps_x
                )
                self.z_hat[t] = (
                    torch.sqrt(self.alpha) * self.f_phi_x(self.x_hat[t:t+1].clone(), t)
                    + torch.sqrt(1 - self.alpha - self.sigma_z ** 2) * self.z_hat[t-1:t]+ eps_z
                )

        return self.z_hat

    def loss(self, a, b, c, z):
        L1 = F.mse_loss(self.x_hat, self.obsrv) * self.sigma_z ** 2
        L2 = F.mse_loss(z[1:], self.z_hat[1:]) * self.sigma_x ** 2
        L = b * L2 + a * L1
        # print(f'L1={L1.item():.6f}, L2={L2.item():.6f}, L={L.item():.6f}')
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

class get_dataset(Dataset):
    def __init__(self, x, z, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.z[index]

class get_dataset_HC(Dataset):
    def __init__(self, x, z, v, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, index):
        return self.x[:, index, :], self.z[:, index, :], self.v[:, index, :]
