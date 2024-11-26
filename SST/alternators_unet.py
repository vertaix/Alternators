
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import einops
import math
from torch.autograd import Variable

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import Module

class Swish(nn.Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, input_dim=32):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.activation =Swish()

        assert self.input_dim in [32,64]

        self.sq_dim = (self.input_dim // 32) * 2
        self.linear_dim = int(self.sq_dim * self.sq_dim * 512)

        self.cnn_enc_1 = nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_enc_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_enc_3 = nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_enc_4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.linear_enc = nn.Linear(self.linear_dim, self.latent_dim )
        self.bn_enc_1 = nn.BatchNorm2d(32)
        self.bn_enc_2 = nn.BatchNorm2d(64)
        self.bn_enc_3 = nn.BatchNorm2d(256)
        self.bn_enc_4 = nn.BatchNorm2d(512)
        self.apply(kaiming_init)

    def forward(self, x):
        out = self.activation(self.bn_enc_1(self.cnn_enc_1(x)))
        out = self.activation(self.bn_enc_2(self.cnn_enc_2(out)))
        out = self.activation(self.bn_enc_3(self.cnn_enc_3(out)))
        out = self.activation(self.bn_enc_4(self.cnn_enc_4(out)))

        out = out.view(-1, self.linear_dim)
        out = self.linear_enc(out)
        return  out

class Decoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, input_dim=32):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.activation = Swish()

        assert self.input_dim in [32,64]

        self.sq_dim = (self.input_dim // 16) * 2
        self.linear_dim = int(self.sq_dim * self.sq_dim * 512)

        self.cnn_dec_1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_dec_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_dec_3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_dec_4 = nn.ConvTranspose2d(64, self.in_channels, kernel_size=3,stride=1,  padding=1, bias=False)
        self.linear_dec = nn.Linear(self.latent_dim, self.linear_dim)
        self.bn_dec_1 = nn.BatchNorm2d(256)
        self.bn_dec_2 = nn.BatchNorm2d(128)
        self.bn_dec_3 = nn.BatchNorm2d(64)

        self.apply(kaiming_init)

    def forward(self, z):
            out = self.linear_dec(z)
            out = out.view(-1, 512, self.sq_dim, self.sq_dim)
            out = self.activation(self.bn_dec_1(self.cnn_dec_1(out)))
            out = self.activation(self.bn_dec_2(self.cnn_dec_2(out)))
            out = self.activation(self.bn_dec_3(self.cnn_dec_3(out)))
            reconstruction = self.cnn_dec_4(out)
            return torch.sigmoid(reconstruction)
class f_phi(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers,alpha, dropout, bidirectional,device):
        super().__init__()
        self.obser_dim = obser_dim
        self.device=device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.alpha = alpha

        self.dropout=dropout

        if bidirectional:
            self.fc_out = nn.Linear(2 * (latent_dim), (latent_dim))
        else:
            self.fc_out = nn.Linear((latent_dim), (latent_dim))

        self.embedding_z = nn.Linear((latent_dim), (latent_dim))
        self.rnn = nn.GRU((latent_dim), (latent_dim), n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, z_x_k, hidden, cell):

        # hidden=self.embedding_z(hidden)
        # hidden = hidden #+ self.sigma_z * torch.randn(hidden.shape).to(self.device)
        z_x_k=z_x_k.unsqueeze(0)

        output, (hidden) = self.rnn(z_x_k, (hidden))

        prediction = self.fc_out(output.squeeze(0))
        # prediction=torch.sqrt(self.alpha)*embedded_new+ torch.sqrt(1-self.alpha)*prediction
        # prediction = [batch size, output dim]

        return prediction, hidden, cell





class Swish(Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def encoder(self, x: torch.Tensor, t: torch.Tensor):
        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)
        return x, t, h

    def decoder(self, x: torch.Tensor, t: torch.Tensor, h):
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        z, t, h = self.encoder(x, t)

        return self.decoder(z, t, h)



class Alt_static_unet(nn.Module):
    def __init__(self, latent_dim, obser_dim, in_channels,noise_max, n_layers, number_step, alpha, device):
        super().__init__()
        self.device = device
        self.latent_dim = torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.number_step = number_step
        self.sigma_x = noise_max+0*torch.linspace(noise_max, 0.001, steps=number_step, requires_grad=False).to(self.device)
        self.alpha_k = .5 + 0 * torch.linspace(0.01, 1 - np.max([noise_max, alpha * noise_max]), steps=number_step,
                                               requires_grad=False).to(self.device)
        self.alpha = torch.tensor([alpha], requires_grad=False).to(self.device)
        self.sigma_z = self.sigma_x  # / self.alpha

        self.n_layers = n_layers
        self.in_channels=in_channels
        self.dp_rate = .1
        self.obser_dim = torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.f_vae = UNet(
            image_channels=self.in_channels,
            n_channels=128,
            ch_mults=[1, 2, 2],
            is_attn=[False, False, False],
        ).to(device)

    def forward(self, obsrv, eps_x_org, eps_z_org, obsr_enable, enabled_noise=True):

        self.obsrv = obsrv.to(self.device)  # .repeat(1,self.importance_sample_size,1)
        batch_size = self.obsrv.shape[1]

        # tensor to store decoder outputs
        self.x_hat = torch.zeros(self.number_step, batch_size, self.in_channels, self.obser_dim, self.obser_dim).to(self.device)

        self.z_hat = torch.zeros(self.number_step, batch_size, eps_z_org.shape[-3], eps_z_org.shape[-2], eps_z_org.shape[-1]).to(
            self.device)
        self.z_x_hat= torch.zeros_like(self.z_hat)
        self.x_inf = torch.zeros(self.number_step, batch_size, self.in_channels, self.obser_dim, self.obser_dim,
                                 requires_grad=False).to(self.device)
        self.z_inf = torch.zeros_like(self.z_hat, requires_grad=False).to(self.device)

        self.z_inf[0] = self.sigma_z[0] *eps_z_org



        self.z_hat[0], t, h = self.f_vae.encoder(self.sigma_x[0] *eps_x_org, 0 * torch.ones(batch_size, ).to(self.device))
        self.x_inf=self.obsrv
        for k in range(1, self.number_step):
        

            eps_z = (self.sigma_z[k] * eps_z_org
                     + self.sigma_z[k] * torch.randn(self.z_hat[0].shape).to(
                        self.device)) / 2
            eps_x = (self.sigma_x[k] * eps_x_org
                     + self.sigma_x[k] * torch.randn(self.x_hat[0].shape).to(
                        self.device)) / 2

            if obsr_enable[k]:

                self.x_hat[k] =  self.f_vae.decoder(
                    torch.sqrt(
                        1 - torch.pow(self.sigma_z[k], 2)) *
                    self.z_hat[
                        k - 1] + eps_z,t,h)

                self.z_x_hat[k], t,h = self.f_vae.encoder(
                    torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.obsrv[k] + eps_x,k*torch.ones(batch_size,).to(self.device))

                self.z_hat[k] = torch.sqrt(self.alpha) * self.z_x_hat[k] + torch.sqrt(1 - self.alpha) * self.z_hat[
                    k - 1]

            else:

                if enabled_noise:

                    self.x_hat[k] = torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.f_vae.decoder(
                    torch.sqrt(
                        1 - torch.pow(self.sigma_z[k], 2)) *
                    self.z_hat[
                        k - 1] + eps_z,t,h) + eps_x

                    temp,t,h=self.f_vae.encoder(
                    torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.x_hat[k] + eps_x,k*torch.ones(batch_size,).to(self.device))
                    self.z_hat[k]= torch.sqrt(self.alpha) * temp + torch.sqrt(1 - self.alpha - torch.pow(self.sigma_z[k], 2)) * self.z_hat[k - 1] + eps_z
                else:
                    self.x_hat[k] =  self.f_vae.decoder(
                        torch.sqrt(
                            1 - torch.pow(self.sigma_z[k], 2)) *
                        self.z_hat[
                            k - 1] + eps_z, t, h)

                    temp, t, h = self.f_vae.encoder(
                        torch.sqrt(1 - torch.pow(self.sigma_x[k], 2)) * self.x_hat[k] + eps_x,
                        k * torch.ones(batch_size, ).to(self.device))
                    self.z_hat[k] = torch.sqrt(self.alpha) * temp + torch.sqrt(
                        1 - self.alpha ) * self.z_hat[k - 1]
        return self.z_hat, self.x_hat

    def loss(self, a, b):

        L1 = torch.mean(
            torch.mean(F.mse_loss(self.x_hat[1:], self.x_inf[1:], reduction='none'), dim=[1, 2,3,4]))
        L2 = torch.mean(torch.mean(F.mse_loss(
            (self.z_hat[1:] - torch.sqrt(1 - self.alpha).reshape([-1,1,1, 1, 1]) * self.z_hat[:-1]),
            torch.sqrt(self.alpha).reshape([-1, 1,1,1, 1]) * self.z_x_hat[
                                                               1:], reduction='none'), dim=[1, 2,3,4]))

        L = b * L2 + a * L1

        return L


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

