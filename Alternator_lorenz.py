import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import pickle

''' data'''

from Lorenz import lorenz_sample_generator
from Alt import *

Lorenz_dataset = pickle.load(open("Lorenz_dataset.p", "rb"))
z = Lorenz_dataset['z'][:]
Spikes = Lorenz_dataset['Spikes'][:]
''' spiking observations'''

Spikes = np.delete(Spikes, np.where(Spikes.sum(axis=0) < 10)[0], axis=1)
Spikes = gaussian_kernel_smoother(Spikes, 2, 6)
# x =(Spikes -np.mean(Spikes,axis=0))/np.std(Spikes,axis=0)
x = calDesignMatrix_V2(Spikes, 1 + 1).squeeze()

x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
z =  (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))

device = torch.device('cpu')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0], shuffle=False)
model = Alt(latent_dim=z.shape[1], obser_dim=x.shape[1],
               sigma_x=.1, alpha=.3,
               importance_sample_size=1, n_layers=2,
               device=device).to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
warmup_epochs=5
Numb_Epochs = 50

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Numb_Epochs - warmup_epochs - 1,
                                                           eta_min=1e-3)
CLIP = 1
total_loss = []

for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        x = torch.unsqueeze(x, 1)
        z = torch.unsqueeze(z, 1)

        optimizer.zero_grad()
        mask_imput = get_mask_forcasting(x.shape[0], 0)
        eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
        eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
        z_hat = model(x, mask_imput, eps_x, eps_z, True)

        print('epoch=%d/%d' % (epoch, Numb_Epochs))
        loss = model.loss(a=10, b=1, c=10, z=z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()


        epoch_loss += loss.item()
    total_loss.append(epoch_loss)
    if epoch > warmup_epochs:
        scheduler.step()


# torch.save(model.state_dict(), 'NODE.pt')
import matplotlib.pyplot as plt

''' visualization'''
for i, batch in enumerate(Dataset_loader):
    x, z = batch
    x = torch.unsqueeze(x, 1)
    z = torch.unsqueeze(z, 1)

x_n = x.detach().cpu().numpy().squeeze()
save_result_path = 'Results/'

z = z.detach().cpu().numpy().squeeze()


f, axes = plt.subplots(3, 1, sharex=True, sharey=False)
plt.close()
ff=plt.figure()
ax = ff.axes(projection='3d')
trj_samples = np.random.randint(0, 1, 20)

r2_tr = []
rho_tr = []
cc_tr = []
mae_tr = []
rmse_tr = []
vendi_tr = []
tru_traj = []
estimaed_traj = []
qq=0
for ii in trj_samples:
    mask_imput = get_mask_forcasting(x.shape[0], 0)
    eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
    eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
    z_hat = model(x, mask_imput, eps_x, eps_z, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, :]

    z_hat = (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0))
    if qq == 0:
        qq+=1
        z_hat_m = z_hat/len(trj_samples)
    else:
        qq += 1
        z_hat_m += z_hat / len(trj_samples)

    axes[0].plot(z_hat[:, 0].squeeze(), 'goldenrod',alpha=3/10)
    axes[1].plot(z_hat[:, 1].squeeze(), 'goldenrod',alpha=3/10)
    axes[2].plot(z_hat[:, 2].squeeze(), 'goldenrod',alpha=3/10)
    tru_traj.append(z[1:, :].squeeze())
    estimaed_traj.append(z_hat.squeeze())
tru_traj = np.array(tru_traj)
estimaed_traj = np.array(estimaed_traj)
axes[0].plot(z_hat_m[:, 0].squeeze(), 'goldenrod')
axes[1].plot(z_hat_m[:, 1].squeeze(), 'goldenrod')
axes[2].plot(z_hat_m[:, 2].squeeze(), 'goldenrod')
axes[0].plot(z[1:, 0].squeeze(), 'k')
axes[1].plot(z[1:, 1].squeeze(), 'k')
axes[2].plot(z[1:, 2].squeeze(), 'k')
plt.title('with observations')

plt.savefig(save_result_path + 'Lorenz-with-obsr.png')
plt.savefig(save_result_path + 'Lorenz-with-obsr.svg', format='svg')
plt.close()

############################
