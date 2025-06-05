import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import *
from Alt import Alt, init_weights
class get_dataset(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, x,z, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.z[index]]
# Load and preprocess Lorenz dataset
Lorenz_dataset = pickle.load(open("Lorenz_dataset.p", "rb"))
z = Lorenz_dataset['z']
Spikes = Lorenz_dataset['Spikes']

# Preprocess spikes: smooth and normalize
Spikes = np.delete(Spikes, np.where(Spikes.sum(axis=0) < 10)[0], axis=1)
Spikes = gaussian_kernel_smoother(Spikes, 2, 6)
x = calDesignMatrix_V2(Spikes, 2).squeeze()

# Normalize to [0, 1]
x =2* (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-6)-1
z = 2*(z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + 1e-6)-1

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoader
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0], shuffle=False)

# Initialize model
model = Alt(
    latent_dim=z.shape[1],
    obser_dim=x.shape[1]*x.shape[2],
    sigma_x=0.2,
    alpha=0.7,
    importance_sample_size=1,
    n_layers=3,
    device=device
).to(device)
model.apply(init_weights)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=45, eta_min=1e-3
# )
CLIP = 1
warmup_epochs = 5
Numb_Epochs = 200
total_loss = []

# Training loop
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x_batch, z_batch = batch
        x_batch=x_batch.reshape([x_batch.shape[0],-1])
        x_batch = x_batch.unsqueeze(1)  # (T, B=1, D_x)
        z_batch = z_batch.unsqueeze(1)  # (T, B=1, D_z)

        optimizer.zero_grad()

        mask_imput = get_mask_imputation(x_batch.shape[0], 20).to(device)
        eps_z = model.sigma_z * torch.randn(z_batch[0].shape).to(device)
        eps_x = model.sigma_x * torch.randn(x_batch[0].shape).to(device)

        _ = model(x_batch, mask_imput, eps_x, eps_z, obsr_enable=True)
        loss = model.loss(a=1, b=1, c=1, z=z_batch)

        print(f'Epoch {epoch + 1}/{Numb_Epochs} - Loss: {loss.item():.6f}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()

    total_loss.append(epoch_loss)
    # if epoch > warmup_epochs:
    #     scheduler.step()

# Save model
os.makedirs("Results", exist_ok=True)
torch.save(model.state_dict(), "Results/Alt_Lorenz_model.pt")

# Evaluation and Visualization
model.eval()
for i, batch in enumerate(Dataset_loader):
    x_eval, z_eval = batch
    x_eval=x_eval.reshape([x_batch.shape[0],-1])
    x_eval = x_eval.unsqueeze(1)
    z_eval = z_eval.unsqueeze(1)

x_np = x_eval.detach().cpu().numpy().squeeze()
z_np = z_eval.detach().cpu().numpy().squeeze()

trj_samples = 10
z_hats = []

for _ in range(trj_samples):
    mask_imput = get_mask_imputation(x_eval.shape[0], 0).to(device)
    eps_z = model.sigma_z * torch.randn(z_eval[0].shape).to(device)
    eps_x = model.sigma_x * torch.randn(x_eval[0].shape).to(device)

    z_hat = model(x_eval, mask_imput, eps_x, eps_z, obsr_enable=True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, :]
    # z_hat = (z_hat - z_hat.min(0)) / (z_hat.max(0) - z_hat.min(0) + 1e-6)
    z_hats.append(z_hat)

z_hats = np.array(z_hats)
z_mean = z_hats.mean(axis=0)
z_true = z_np[1:]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
colors = ['goldenrod'] * 3
labels = ['z1', 'z2', 'z3']

for i in range(3):
    for traj in z_hats:
        axes[i].plot(traj[:, i], color=colors[i], alpha=0.3)
    axes[i].plot(z_mean[:, i], color=colors[i], label='Mean Predicted')
    axes[i].plot(z_true[:, i], color='k', label='Ground Truth')
    axes[i].set_ylabel(labels[i])
    if i == 0:
        axes[i].legend()

axes[-1].set_xlabel('Time step')
plt.suptitle('Latent Trajectory Prediction with Observations (Lorenz)')
plt.tight_layout()
plt.savefig("Results/Lorenz-with-obsr.png")
plt.savefig("Results/Lorenz-with-obsr.svg", format='svg')
plt.close()

print("Evaluation plots saved to Results/")

# Save evaluation results
results = {
    'z_true': z_true,
    'z_hats': z_hats,
    'z_mean': z_mean,
    'MAE': np.abs(z_mean - z_true).mean(),
    'MSE': ((z_mean - z_true) ** 2).mean(),
}
print(results)
with open("Results/Alt_Lorenz_results.p", "wb") as f:
    pickle.dump(results, f)
