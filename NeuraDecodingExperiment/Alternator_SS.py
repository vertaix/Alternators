import pickle

import numpy as np
import torch
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from utils import *

from LDIP_A_Att import *
dataset = pickle.load(open("data/NeuralData/example_data_s1_pp.p", "rb"))
batch_size=3
''' data'''


def preprocess_HC(x_in,z_in,downsample_rate, episod_len):
    x = calSmoothNeuralActivity(np.squeeze(x_in), 10, 5)
    z = z_in
    data_len = x.shape[0]

    ''' normalization'''
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    z = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))

    ''' down sampling rate'''

    x = x[np.arange(0, data_len, downsample_rate), :]
    x = calDesignMatrix_V2(x, 3).squeeze()
    z = z[np.arange(0, data_len, downsample_rate), :]

    x_new = np.zeros((episod_len,(x.shape[0]//episod_len), x.shape[1]))
    z_new = np.zeros((episod_len, (x.shape[0]//episod_len), z.shape[1]))
    for ii in range(x.shape[0]//episod_len):
        x_new[:, ii, :] = x[ii*episod_len:(ii+1)*episod_len, :]
        z_new[:, ii, :] = z[ii*episod_len:(ii+1)*episod_len, :]
    return x_new,z_new
[x_tr,z_tr]= preprocess_HC(dataset['X_train'],dataset['Y_train'][:, :2], 5,200)
[x_val,z_val]= preprocess_HC(dataset['X_val'],dataset['Y_val'][:, :2], 5,200)
[x_test,z_test]= preprocess_HC(dataset['X_test'],dataset['Y_test'][:, :2], 5,200)
''''''
device = torch.device('cpu')
Dataset = get_dataset_HC(x_tr, z_tr, device)
Dataset_val = get_dataset_HC(x_val, z_val, device)
Dataset_test = get_dataset_HC(x_test, z_test, device)
Dataset_loader = DataLoader(Dataset, batch_size=batch_size,shuffle=False)
Dataset_val_loader = DataLoader(Dataset_val, batch_size=z_val.shape[1],shuffle=False)
Dataset_test_loader = DataLoader(Dataset_test, batch_size=z_test.shape[1],shuffle=False)
model = LDIP_A(latent_dim=z_tr.shape[-1], obser_dim=x_tr.shape[-1], sigma_x=.2,alpha=.3, importance_sample_size=1, n_layers=2,
              device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
# print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
# model.load_state_dict(torch.load('LDIP_D_SS_5mc_LSTM_5L_V7.pt'))
''' training phase'''
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss=[]
Numb_Epochs=500
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        x= torch.swapaxes(x, 0,1)
        z = torch.swapaxes(z, 0,1)

        optimizer.zero_grad()
        mask_imput = get_mask_forcasting(x.shape[0], 0)
        eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
        eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
        z_hat = model(x, mask_imput, eps_x, eps_z, True)
        print('epoch=%d/%d'%(epoch,Numb_Epochs))
        loss=model.loss(a=100,b=1,c=100,z=z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()
    total_loss.append(epoch_loss)

''' save and load models'''
torch.save(model.state_dict(), 'LDIP_D_SS_5mc_LSTM_5L_V7.pt')
#

''''''

import matplotlib.pyplot as plt

''' visualization'''
from metrics import  get_R2, get_rho, get_RMSE, get_MAE,get_pearsonr
max_numb_repret=15
save_result_path = 'Results/'
# plt.figure()
# plt.plot(total_loss)
# plt.show()

# Dataset_loader = DataLoader(Dataset, batch_size=z_tr.shape[1],shuffle=False)
# for i, batch in enumerate(Dataset_loader):
#     x, z = batch
#     x = torch.swapaxes(x, 0, 1)
#     z = torch.swapaxes(z, 0, 1)
#
# z = z.detach().cpu().numpy().squeeze()
#
# trj_samples = np.arange(0, z.shape[1])
# r2_tr=[]
# rho_tr=[]
# cc_tr=[]
# mae_tr=[]
# rmse_tr=[]
# for ii in trj_samples:
#     f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
#     for num_rep in range(max_numb_repret):
#         mask_imput = get_mask_forcasting(x.shape[0], 0)
#         eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
#         eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
#         z_hat = model(x, mask_imput, eps_x, eps_z, True)
#         z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]
#
#         z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
#
#
#         axes[0].plot(z_hat[:, 0].squeeze(), 'r')
#         axes[1].plot(z_hat[:, 1].squeeze(), 'r')
#     axes[0].plot(z[1:, ii, 0].squeeze(), 'k')
#     axes[1].plot(z[1:,ii, 1].squeeze(), 'k')
#     plt.title('with observations')
#     plt.savefig(save_result_path + 'Train-'+str(ii)+'-SS-with-obsr.png')
#     plt.savefig(save_result_path + 'Train-'+str(ii)+'-SS-with-obsr.svg', format='svg')
#     rho_tr.append(np.mean(np.abs(get_rho(z[1:, ii, :].squeeze(), z_hat.squeeze()))))
#     r2_tr.append(np.mean(np.abs(get_R2(z[1:, ii, :].squeeze(), z_hat.squeeze()))))
#     cc_tr.append(np.mean(np.abs(get_pearsonr(z[1:, ii, :].squeeze(), z_hat.squeeze()))))
#     mae_tr.append(np.mean(np.abs(get_MAE(z[1:, ii, :].squeeze(), z_hat.squeeze()))))
#     rmse_tr.append(np.mean(np.abs(get_RMSE(z[1:, ii, :].squeeze(), z_hat.squeeze()))))
# print('train rho-mean=%f, std=%f' % (np.mean(rho_tr), np.std(rho_tr)))
# print('train R2-mean=%f, std=%f' % (np.mean(r2_tr), np.std(r2_tr)))
# print('train CC-mean=%f, std=%f' % (np.mean(cc_tr), np.std(cc_tr)))
# print('train MAE-mean=%f, std=%f' % (np.mean(mae_tr), np.std(mae_tr)))
# print('train RMSE-mean=%f, std=%f' % (np.mean(rmse_tr), np.std(rmse_tr)))

""" validation result """
# r2_val = []
# rho_val = []
# cc_val = []
# mae_val = []
# rmse_val = []
# for i, batch in enumerate(Dataset_val_loader):
#     x, z = batch
#     x = torch.swapaxes(x, 0, 1)
# z = z.detach().cpu().numpy().squeeze()
#
# trj_samples = np.arange(z_val.shape[1])
# for ii in trj_samples:
#     f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
#     for num_rep in range(max_numb_repret):
#         mask_imput = get_mask_forcasting(x.shape[0], 0)
#         eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
#         eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
#         z_hat = model(x, mask_imput, eps_x, eps_z, True)
#         z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]
#
#         z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
#
#         axes[0].plot(z_hat[:, 0].squeeze(), 'r')
#         axes[1].plot(z_hat[:, 1].squeeze(), 'r')
#     axes[0].plot(z[ii,1:, 0].squeeze(), 'k')
#     axes[1].plot(z[ii,1:, 1].squeeze(), 'k')
#
#     plt.title('with  observations')
#     plt.savefig(save_result_path + 'Val-'+str(ii)+'-SS-with-obsr.png')
#     plt.savefig(save_result_path + 'Val-'+str(ii)+'-SS-with-obsr.svg', format='svg')
#     rho_val.append(np.mean(np.abs(get_rho(z[ii, 1:, :].squeeze(), z_hat.squeeze()))))
#     r2_val.append(np.mean(np.abs(get_R2(z[ii, 1:, :].squeeze(), z_hat.squeeze()))))
#     cc_val.append(np.mean(np.abs(get_pearsonr(z[ii, 1:, :].squeeze(), z_hat.squeeze()))))
#     mae_val.append(np.mean(np.abs(get_MAE(z[ii, 1:, :].squeeze(), z_hat.squeeze()))))
#     rmse_val.append(np.mean(np.abs(get_RMSE(z[ii, 1:, :].squeeze(), z_hat.squeeze()))))
# print('val rho-mean=%f, std=%f' % (np.mean(rho_val), np.std(rho_val)))
# print('val R2-mean=%f, std=%f' % (np.mean(r2_val), np.std(r2_val)))
# print('val CC-mean=%f, std=%f' % (np.mean(cc_val), np.std(cc_val)))
# print('val MAE-mean=%f, std=%f' % (np.mean(mae_val), np.std(mae_val)))
# print('val RMSE-mean=%f, std=%f' % (np.mean(rmse_val), np.std(rmse_val)))

""" Test result """
r2_te = []
rho_te = []
cc_te = []
mae_te = []
rmse_te = []
for i, batch in enumerate(Dataset_test_loader):
    x, z = batch
    z = torch.swapaxes(z, 0, 1)
    x = torch.swapaxes(x, 0, 1)
z = z.detach().cpu().numpy().squeeze()

trj_samples = np.arange(z_val.shape[1])
for ii in trj_samples:
    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    for num_rep in range(max_numb_repret):
        mask_imput = get_mask_forcasting(x.shape[0], 0)
        eps_z = model.sigma_z * torch.randn((z.shape[1],z.shape[2])).to(device)
        eps_x = 1 * model.sigma_x * torch.randn(x[0].shape).to(device)
        z_hat = model(x, mask_imput, eps_x, eps_z, True)
        z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]

        z_hat =(z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0))

        axes[0].plot(z_hat[:, 0].squeeze(), 'r')
        axes[1].plot(z_hat[:, 1].squeeze(), 'r')
    axes[0].plot(z[1:,ii, 0].squeeze(), 'k')
    axes[1].plot(z[1:,ii, 1].squeeze(), 'k')

    plt.title('with  observations')
    plt.savefig(save_result_path + 'Test-'+str(ii)+'-SS-with-obsr.png')
    plt.savefig(save_result_path + 'Test-'+str(ii)+'-SS-with-obsr.svg', format='svg')

    rho_te.append(np.mean(np.abs(get_rho(z[1:,ii, :].squeeze(), z_hat.squeeze()))))
    r2_te.append(np.mean(np.abs(get_R2(z[1:,ii, :].squeeze(), z_hat.squeeze()))))
    cc_te.append(np.mean(np.abs(get_pearsonr(z[1:,ii, :].squeeze(), z_hat.squeeze()))))
    mae_te.append(np.mean(np.abs(get_MAE(z[1:,ii, :].squeeze(), z_hat.squeeze()))))
    rmse_te.append(np.mean(np.abs(get_RMSE(z[1:,ii, :].squeeze(), z_hat.squeeze()))))
print('test rho-mean=%f, std=%f' % (np.mean(rho_te), np.std(rho_te)))
print('test R2-mean=%f, std=%f' % (np.mean(r2_te), np.std(r2_te)))
print('test CC-mean=%f, std=%f' % (np.mean(cc_te), np.std(cc_te)))
print('test MAE-mean=%f, std=%f' % (np.mean(mae_te), np.std(mae_te)))
print('test RMSE-mean=%f, std=%f' % (np.mean(rmse_te), np.std(rmse_te)))
