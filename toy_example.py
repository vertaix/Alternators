import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from alternators_static import *
import lib.toy_data as toy_data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
scaler = MinMaxScaler()
def main():
    parser = argparse.ArgumentParser(description="toy example")
    # Add arguments to the parser with predefined default values
    parser.add_argument('--dataset_name',  type=str, default='pinwheel', help="name of the dataset, checkerboard, 2spirals, 8gaussians, pinwheel")
    parser.add_argument('--test',  type=bool, default=True, help="True if you want to use pretrained model otherwise it should be False for training ")
    parser.add_argument('--device',  type=str, default='cpu', help=" 'cpu', or 'cuda' ")
    parser.add_argument('--batch_size', type=int, default=1000, help="batch_size")
    parser.add_argument('--num_epochs',  type=int, default=10000, help="num_epochs")
    parser.add_argument('--N_steps',  type=int, default=15, help="N_steps")
    parser.add_argument('--latent_size',  type=int, default=20, help="latent_size")
    parser.add_argument('--noise_max', type=float, default=0.5, help="noise_max")
    parser.add_argument('--alpha',  type=float, default=0.7, help="alpha")
    parser.add_argument('--n_layers', type=int, default=1, help="number of the attention layer")
    parser.add_argument('--n_samples',  type=int, default=20000, help="number of samples for generation ")
    parser.add_argument('--model_path',  type=str, default='./pretrained_models/', help="path to the folder for the pretrained models ")
    parser.add_argument('--result_path', type=str, default='./toy_gif/',
                        help="path to the folder for saving ")

    # Parse the command-line arguments
    args = parser.parse_args()

    dataset = toy_data.inf_train_gen(args.dataset_name, batch_size=10000)
    dataset = 4 * scaler.fit_transform(dataset) - 2
    # Hyperparameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = torch.device(args.device)
    test = args.test
    train_dataset = TensorDataset(torch.Tensor(dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    N_steps = args.N_steps
    latent_size = args.latent_size
    model = Alt_static(latent_dim=latent_size, obser_dim=2, number_step=N_steps, alpha=args.alpha,
                      noise_max=args.noise_max, n_layers=args.n_layers,
                      device=device).to(device)

    ll_best = 1e14
    if test:
        model.load_state_dict(
            torch.load(args.model_path+'Alt_toy' + args.dataset_name + '_latent_' + str(latent_size) + '_N_' + str(N_steps) + '.pt',
                       map_location=torch.device('cpu')))


    else:
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1e-3, weight_decay=1e-6)
        total_loss = []
        step = 0
        for epoch in range(num_epochs):
            running_loss = 0.0
            epoch_loss = 0
            for i, data in enumerate(train_loader, 0):
                inputs = data[0]

                optimizer.zero_grad()
                eps_z0 = torch.randn([inputs.shape[0], latent_size]).to(device)
                eps_x0 = torch.randn(inputs.shape).to(device)
                z_hat, x_hat = model(inputs, eps_x0, eps_z0, True)
                L = model.loss(1, 1)
                L.backward()
                optimizer.step()
                epoch_loss += L.item()
                step += 1

            print('epoch=%d-%d' % (epoch, num_epochs))
            print('LL=%f, best-LL=%f' % (epoch_loss, ll_best))
            if (epoch_loss < ll_best):
                ll_best = epoch_loss
                torch.save(model.state_dict(),
                           args.model_path+'Alt_toy' + args.dataset_name + '_latent_' + str(latent_size) + '_N_' + str(N_steps) + '.pt')
            total_loss.append(epoch_loss)

    test_dataset = train_dataset
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=dataset.shape[0], shuffle=True)
    # Visualize a few images
    for kk in range(1, N_steps):
        # threshold
        gen_sampels = []
        real_samples = []
        true_data = []
        for iii in range(args.n_samples//batch_size):

            model.eval()
            for i, (image) in enumerate(test_loader, 1):
                image = image[0]
                eps_z0 = torch.randn([image.shape[0], latent_size]).to(device)
                eps_x0 = torch.randn(image.shape).to(device)
                _, _ = model(eps_x0, eps_x0, eps_z0, False, True)
                x_hat = model.x_hat.detach().cpu().squeeze()[kk]
                gen_sampels.append(x_hat.squeeze())
                real_samples.append(image.squeeze())

        gen_sampels = torch.cat(gen_sampels, dim=0)
        real_samples = torch.cat(real_samples, dim=0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        plt_flow_samples(gen_sampels, axes[0], npts=150, memory=50000, kde_enable=False,
                         title="generated_samples_by_alternator", device="cpu")
        plt_flow_samples(real_samples, axes[1], npts=150, memory=50000, kde_enable=False, title="data", device="cpu")
        plt.tight_layout()
        axes[0].set_title('Alternator ', size=24)
        axes[1].set_title('Data', size=24)
        fig.subplots_adjust(left=0.0, right=1, top=.9, bottom=0.0)
        if  args.dataset_name == '2spirals':  # , '8gaussians', 'checkerboard, moons, pinwheel, 2spirals,
            bias_d = 10000
        elif  args.dataset_name == 'pinwheel':
            bias_d = 20000
        elif  args.dataset_name == 'checkerboard':
            bias_d = 30000
        elif  args.dataset_name == '8gaussians':
            bias_d = 40000
        elif  args.dataset_name == 'moons':
            bias_d = 50000

        formatted_number = "{:0>{length}}".format(bias_d + kk, length=7)
        plt.savefig(args.result_path + 'image_' + args.dataset_name + '_' + formatted_number + '.png')


if __name__ == '__main__':
    main()

