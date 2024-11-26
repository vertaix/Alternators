import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from alternators_unet import *
from utils import add_sn
import pickle
# Hyperparameters
batch_size =10
num_epochs = 400
device = torch.device('cuda')
warmup_epochs=5
test=False

with open('SST_dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

class get_dataset(torch.utils.data.Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, x, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:,:,:]

x_tr=dataset['x_tr']
train_dataset = get_dataset(x_tr, device)
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False)

N_steps=dataset['N_steps']
latent_size=1024
noise_max=0.1
model = Alt_static_unet(latent_dim=latent_size, obser_dim=60,n_layers=1, number_step=N_steps,alpha=.7,
              noise_max=noise_max, in_channels=1,
              device=device).to(device)
model.apply(init_weights)
model.apply(add_sn)


ll_best=1e10

print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
print('latent_size=%f'%(latent_size))
print('N_steps=%f'%(N_steps))
print('noise_max=%f'%(noise_max))
l1_loss=torch.nn.L1Loss()
if test:
    model.load_state_dict(torch.load('Alt_SST_' + str(latent_size) + '_N_' + str(N_steps)+'-noise-max-'+str(noise_max) + '.pt',
                                     map_location=torch.device('cpu')))

else:
    model.load_state_dict(torch.load('Alt_SST_' + str(latent_size) + '_N_' + str(N_steps)+'-noise-max-'+str(noise_max) + '.pt',map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-3,weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=num_epochs-warmup_epochs-1,
                                                           eta_min=1e-4)
    total_loss=[]
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            
            data=torch.unsqueeze(data, dim=2)
            inputs=torch.transpose(data, dim0=0, dim1=1)
            optimizer.zero_grad()
            eps_z0 = torch.randn([inputs.shape[1],  512,15,15]).to(device)
            eps_x0 = torch.randn(inputs[0].shape).to(device)
            z_hat, x_hat = model(inputs, eps_x0, eps_z0, torch.ones((N_steps,),dtype=torch.bool,requires_grad=False))
            L = model.loss(1, 1)
            # print('L1=%f, L2=%f, L=%f' % (L1, L2, L))
            L.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += L.item()
        print('LL=%f,best_model=%f'%(epoch_loss,ll_best))
        print('epoch=%d-%d'%(epoch,num_epochs))
        if epoch > warmup_epochs:
            scheduler.step()
        if (epoch_loss < ll_best):
            ll_best = epoch_loss
            torch.save(model.state_dict(), 'Alt_SST_' + str(latent_size) + '_N_' + str(N_steps)+'-noise-max-'+str(noise_max) + '.pt')
        total_loss.append(epoch_loss)

