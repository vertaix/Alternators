import pickle

import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from utils import *
from Alt import *

import numpy as np

with open('dataset_simple.p', 'rb') as file:
    dataset = pickle.load(file)
xs=dataset['xs']
vs=dataset['vendis']
zs=dataset['latents']


xs=np.expand_dims(xs, axis=0)
zs=np.expand_dims(zs, axis=0)
vs=np.expand_dims(vs, axis=0)

device = torch.device('cpu')
Dataset_tr = get_dataset_HC(xs, zs,vs, device)
Dataset_loader_tr = DataLoader(Dataset_tr, batch_size=zs.shape[1],shuffle=False)
model = Alt(latent_dim=1, obser_dim=xs.shape[-1], sigma_x=.4, alpha=.7,
               importance_sample_size=1, n_layers=2,
               device=device).to(device)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
# print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The Alt model has {count_parameters(model):,} trainable parameters')
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
CLIP = 1
total_loss=[]
Numb_Epochs=500
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader_tr):
        x, z,v = batch


        optimizer.zero_grad()
        mask_imput = get_mask_forcasting(x.shape[0], 0)

        eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
        eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
        z_hat = model(x, mask_imput, eps_x, eps_z, obsr_enable=True, smoother_enable=False)
        print('epoch=%d/%d'%(epoch,Numb_Epochs))
        loss=model.loss(a=1,b=1,c=1,z=z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()
    total_loss.append(epoch_loss)

torch.save(model.state_dict(), 'DD_model.pt')
#
model.load_state_dict(torch.load('DD_model.pt', map_location=torch.device('cpu')))

import matplotlib.pyplot as plt

''' visualization'''
with open('dataset_simple.p', 'rb') as file:
    dataset = pickle.load(file)
xs_te=dataset['xs']
vs_te=dataset['vendis']
zs_te=dataset['latents']


xs_te=np.expand_dims(xs_te, axis=0)
vs_te=np.expand_dims(vs_te, axis=0)
zs_te=np.expand_dims(zs_te, axis=0)
Dataset_te = get_dataset_HC(xs_te, zs_te,vs_te, device)
Dataset_loader_te = DataLoader(Dataset_te, batch_size=zs_te.shape[1],shuffle=False)

for i, batch in enumerate(Dataset_loader_te):
    x, z,v = batch
z = z.detach().cpu().numpy().squeeze()

trj_samples=np.arange(0,10)

all_z_hats=[]
all_x_hats=[]
for ii in trj_samples:
    mask_imput = get_mask_forcasting(x.shape[0], 0)

    eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
    eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
    z_hat = model(x, mask_imput, eps_x, eps_z, obsr_enable=False, smoother_enable=False)
    all_z_hats.append( z_hat.detach().cpu().numpy().squeeze())
    all_x_hats.append(z_hat.detach().cpu().numpy().squeeze())
all_z_hats=np.array(all_z_hats)
all_x_hats=np.array(all_x_hats)


plt.figure(figsize=(14, 4))
plt.plot(z, 'k', linewidth=2)
plt.plot(all_z_hats.T, color='#eb6200', alpha=.3, linewidth=1)
plt.plot(all_z_hats.T.mean(axis=-1), color='#eb6200', alpha=.1, linewidth=2)
plt.title('alternator')
plt.tight_layout()
plt.savefig('alt_simulation.pdf', format='pdf')
plt.show()
print(f'MAE={np.abs(all_z_hats.T.mean(axis=-1)-z).mean()}')
print(f'MSE={((all_z_hats.T.mean(axis=-1)-z)**2).mean()}')
pearson_correlation, _ = pearsonr(all_z_hats.T.mean(axis=-1),z)
print(f'CC={pearson_correlation*100}')

Alt_r= {'all_z_hats': all_z_hats,
             'z': z,
             'pearson_correlation':pearson_correlation,
             'MAE':np.abs(all_z_hats.T.mean(axis=-1)-z).mean(),
             'MSE':((all_z_hats.T.mean(axis=-1)-z)**2).mean()}


pickle.dump(Alt_r, open("alt_toy.p", "wb"))