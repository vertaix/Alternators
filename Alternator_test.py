import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils import get_mask_imputation,get_mask_forcasting
from Alt import Alt, get_dataset_HC

# Load dataset
with open('dataset_simple.p', 'rb') as file:
    dataset = pickle.load(file)
xs = dataset['xs']
vs = dataset['vendis']
zs = dataset['latents']

# Expand dimensions for time-major format
xs = np.expand_dims(xs, axis=0)
zs = np.expand_dims(zs, axis=0)
vs = np.expand_dims(vs, axis=0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader
Dataset_tr = get_dataset_HC(xs, zs, vs, device)
Dataset_loader_tr = DataLoader(Dataset_tr, batch_size=zs.shape[1], shuffle=False)

# Initialize model
model = Alt(latent_dim=1, obser_dim=xs.shape[-1], sigma_x=0.2, alpha=0.7,
            importance_sample_size=1, n_layers=2, device=device).to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
CLIP = 1
total_loss = []
Numb_Epochs = 100

# Training loop
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader_tr):
        x, z, _ = batch
        optimizer.zero_grad()

        mask_imput = get_mask_imputation(x.shape[0], 20).to(device)
        eps_z = model.sigma_z * torch.randn(z.shape[1], z.shape[2]).to(device)
        eps_x = model.sigma_x * torch.randn(x.shape[1], x.shape[2]).to(device)

        _ = model(x, mask_imput, eps_x, eps_z, obsr_enable=True, smoother_enable=False)
        loss = model.loss(a=1, b=1, c=1, z=z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()
    if epoch %5 == 0:
        print(f'Epoch [{epoch+1}/{Numb_Epochs}] - Loss: {epoch_loss:.6f}')

    total_loss.append(epoch_loss)

# Save model
torch.save(model.state_dict(), 'Alt_model.pt')

# Evaluation
model.load_state_dict(torch.load('Alt_model.pt', map_location=device))

# Reload data for test
with open('dataset_simple.p', 'rb') as file:
    dataset = pickle.load(file)
xs_te = dataset['xs']
vs_te = dataset['vendis']
zs_te = dataset['latents']

xs_te = np.expand_dims(xs_te, axis=0)
zs_te = np.expand_dims(zs_te, axis=0)
vs_te = np.expand_dims(vs_te, axis=0)

Dataset_te = get_dataset_HC(xs_te, zs_te, vs_te, device)
Dataset_loader_te = DataLoader(Dataset_te, batch_size=zs_te.shape[1], shuffle=False)

for i, batch in enumerate(Dataset_loader_te):
    x, z, _ = batch
z = z.detach().cpu().numpy().squeeze()

trj_samples = np.arange(0, 10)
all_z_hats = []

for _ in trj_samples:
    mask_imput = get_mask_imputation(x.shape[0], 90).to(device)
    eps_z = model.sigma_z * torch.randn(x.shape[1], model.latent_dim).to(device)
    eps_x = model.sigma_x * torch.randn(x.shape[1], x.shape[2]).to(device)
    z_hat = model(x, mask_imput, eps_x, eps_z, obsr_enable=True, smoother_enable=False)
    all_z_hats.append(z_hat.detach().cpu().numpy().squeeze())

all_z_hats = np.array(all_z_hats)  # [N_samples, T]
z_mean = all_z_hats.mean(axis=0)

# Plot
plt.figure(figsize=(14, 4))
plt.plot(z, 'k', linewidth=2, label='True z')
plt.plot(all_z_hats.T, color='#eb6200', alpha=0.2, linewidth=2)
plt.plot(z_mean, color='#eb6200', alpha=1.0, linewidth=2, label='Mean Predicted z')
plt.title('Alternator Inference')
plt.tight_layout()
plt.legend()
plt.savefig('alt_simulation.pdf', format='pdf')
plt.show()

# Evaluation metrics
MAE = np.abs(z_mean - z).mean()
MSE = ((z_mean - z) ** 2).mean()
CC, _ = pearsonr(z_mean, z)

print(f'MAE={MAE:.6f}')
print(f'MSE={MSE:.6f}')
print(f'CC={CC*100:.2f}')

Alt_r = {
    'all_z_hats': all_z_hats,
    'z': z,
    'pearson_correlation': CC,
    'MAE': MAE,
    'MSE': MSE,
}

with open("Alt_toy.p", "wb") as f:
    pickle.dump(Alt_r, f)
