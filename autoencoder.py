import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import DataLoader
from .database_toy import LightSourceDB
    

''' ----------------------  Autoencoder architecture  ----------------------'''

class VAE(nn.Module):

    def __init__(self, in_chan=1, out_chan=1, latent_chan=8):
        super().__init__()

        # encoder stack - use stride=2 to downsample to [B, 64, H/4, W/4]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # latent space
        self.mu = nn.Conv2d(64, latent_chan, kernel_size=3, stride=1, padding=1)
        self.logvar = nn.Conv2d(64, latent_chan, kernel_size=3, stride=1, padding=1)
        
        # decoder stack
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_chan, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_chan, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid(),  # rescale output [0,1]
        )

    def encode(self, x):
        y = self.encoder(x)
        mu = self.mu(y)
        logvar = self.logvar(y)
        return mu, logvar
    
    # sample from gauss ~ N(0, 1) then scale and shift
    # using mean/var allows process to be stochastic and differentiable
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        gauss = torch.randn_like(std)
        z = mu + gauss * std
        return z

    def forward(self, x):                
        mu, var = self.encode(x)           # mu is deterministic encoding    
        z = self.reparameterize(mu, var)   # z is stochastic encoding
        recon = self.decoder(z)        
        return recon, mu, var, z            # return reconstrution and latent
    

class AutoEncoder(nn.Module):

    def __init__(self, in_chan=1, out_chan=1, latent_chan=8):
        super().__init__()

        # encoder stack
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_chan, kernel_size=3, stride=1, padding=1),
        )
        
        # decoder stack
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_chan, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, out_chan, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # rescale output within [0, 1]
        )

    def forward(self, x):               # [B, 1, H, W]
        latent = self.encoder(x)        # [B, 8, H/4, W/4]
        recon = self.decoder(latent)    # [B, 1, H, W]
        return recon, latent            # return reconstrution and latent
    

''' ----------------------  Training VAE  ----------------------'''

def vae_loss(recon, signal, mu, logvar, loss_fn, beta=1.0):
    recon_loss = loss_fn(recon, signal) # MSE
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def training_looppp(model, dataloader, loss_fn, optimizer, device, data_type, beta):

    model.train()
    train_loss = 0

    # iterate through dataloader by batch. # anat_img_slice, dw_img_slice, bval, bvec
    for input, target, _ in dataloader:

        # either create autoencoder for blob or shadow data
        signal = input.to(device) if data_type == "blob" else target.to(device)

        # forward pass
        #pred, _ = model(signal)
        recon, mu, logvar, z = model(signal)

        # calculate loss between predicted and input
        loss = vae_loss(recon, signal, mu, logvar, loss_fn, beta=beta)
        train_loss += loss.item()

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)  # average loss per batch


def train_vae(model, device, train_loader, loss_fn, optimizer, epochs, batch_size, learning_rate, data_type="blob"):

    # cur model numnber
    model_num = len([f for f in os.listdir('models_diffusion')]) + 1
    model_path = f"models_diffusion/model{model_num}.pth"

    # pandas df to store train/test loss for plotting
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss'])

    # best loss initialized at inf
    best_loss = np.inf

    # iteratively train and test
    for epoch in tqdm(range(epochs)):

        # KL annealing so it doesnt dominate early on
        kl_warmup_epochs = 10
        beta = min(1.0, (epoch + 1) / kl_warmup_epochs)

        train_loss = training_loop(model, train_loader, loss_fn, optimizer, device, data_type, beta)

        # add losses to log
        loss_df.loc[epoch] = [epoch + 1, train_loss]

        # save model with lowest train loss (also store meta data)
        if train_loss < best_loss:
            torch.save({
                'epoch': epoch + 1,
                'batch_size': batch_size,
                'num_samples': len(train_loader.dataset),
                'learning_rate': learning_rate,
                'model_state_dict': model.state_dict(),
                }, model_path)
        
        if epoch % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.5f}")

    # display loss curve
    plot_loss_curve(loss_df)


def plot_loss_curve(loss_df):

    plt.figure(figsize=(8, 5))
    plt.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.title('Training Loss per Epoch')
    plt.yscale('log')  # plotted on log scale to enhance small diff
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()



''' ----------------------  Training autoencoder  ----------------------'''

def training_loop(model, dataloader, loss_fn, optimizer, device, data_type):

    model.train()
    train_loss = 0

    # iterate through dataloader by batch. # anat_img_slice, dw_img_slice, bval, bvec
    for input, target, _ in dataloader:

        # either create autoencoder for blob or shadow data
        signal = input.to(device) if data_type == "blob" else target.to(device)

        # forward pass
        pred, _ = model(signal)
        #recon, mu, var, z = model(signal)

        # calculate loss between predicted and input
        loss = loss_fn(pred, signal)
        train_loss += loss.item()

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)  # average loss per batch


def train_ae(model, device, train_loader, loss_fn, optimizer, epochs, batch_size, learning_rate, data_type="blob"):

    # cur model numnber
    model_num = len([f for f in os.listdir('models_diffusion')]) + 1
    model_path = f"models_diffusion/model{model_num}.pth"

    # pandas df to store train/test loss for plotting
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss'])

    # best loss initialized at inf
    best_loss = np.inf

    # iteratively train and test
    for epoch in tqdm(range(epochs)):

        train_loss = training_loop(model, train_loader, loss_fn, optimizer, device, data_type)

        # add losses to log
        loss_df.loc[epoch] = [epoch + 1, train_loss]

        # save model with lowest train loss (also store meta data)
        if train_loss < best_loss:
            torch.save({
                'epoch': epoch + 1,
                'batch_size': batch_size,
                'num_samples': len(train_loader.dataset),
                'learning_rate': learning_rate,
                'model_state_dict': model.state_dict(),
                }, model_path)
        
        if epoch % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.5f}")

    # display loss curve
    plot_loss_curve(loss_df)


def plot_loss_curve(loss_df):

    plt.figure(figsize=(8, 5))
    plt.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.title('Training Loss per Epoch')
    plt.yscale('log')  # plotted on log scale to enhance small diff
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


''' ----------------------  Testing autoencoder  ----------------------'''

def load_ae(model_path, device):

    # load saved model weights (only saved model weights)
    model_dict = torch.load(model_path, map_location=device)

    # create model and set to eval mode
    loaded_model = AutoEncoder()
    #loaded_model = VAE()
    loaded_model.load_state_dict(model_dict['model_state_dict'])
    loaded_model.to(device)
    loaded_model.eval()

    print_metadata(model_dict)
    
    return loaded_model


def print_metadata(model_dict):

    print(f"Model Metadata:")
    print("-" * 30)
    print(f"  Epochs:          {model_dict['epoch']}")
    print(f"  Learning Rate:  {model_dict['learning_rate']}")
    print(f"  Batch Size:     {model_dict['batch_size']}")
    print(f"  Num Samples:    {model_dict['num_samples']}")


#from src_diffusion.architecture import UNet
#from src_diffusion.database import MRImagesDB
#from src_diffusion.training import sample_batch, train_model, get_noise_scheduler
#from src_diffusion.testing import load_model, sample
#import torchio as tio


def test_model(loaded_model, device, num_samples, num_workers, super_title="", data_type="blob"):

    # load files
    data_dir_path = '/cs/student/projects3/cgvi/2024/morrison/DWsynth_project/'
    bvals_path = data_dir_path + 'bvals_round.bval'
    bvecs_path = data_dir_path + 'bvecs.bvec'
    img_dir_path = data_dir_path + 'test'

    # volume dimensions  [1, H, W, D]
    volume_dims = tio.ScalarImage(data_dir_path + 'train/sub-051-01/anat/sub-051-01_t1.nii.gz').data.shape

    # choose axis to slice along: 0 = saggital, 1 = coronal, 2 = horizontal
    slice_axis = 2


    # create dataset and dataloader
    test_set = MRImagesDB(img_dir_path, bvals_path, bvecs_path, volume_dims, num_samples=num_samples, slice_axis=slice_axis)
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                                num_workers=num_workers,
                                pin_memory=torch.cuda.is_available(),  # speeds up GPU transfer
                                persistent_workers=True)


    # compute predictions
    #test_set = LightSourceDB(num_samples=5)
    #test_loader = DataLoader(dataset=test_set, batch_size=1)

    # store images for display
    inputs, preds = [], []

    loaded_model.eval()
    with torch.no_grad():  # disable grad during inference
        for input, target, _ in test_loader:

            # either create autoencoder for blob or shadow data
            signal = input.to(device) if data_type == "blob" else target.to(device)

            # forward pass
            pred, _ = loaded_model(signal)
            #recon, mu, logvar, z = loaded_model(signal)
            #pred = recon

            inputs.append(signal.squeeze().cpu().numpy())
            preds.append(pred.detach().squeeze().cpu().numpy())

    display_results(inputs, preds, super_title)

    return inputs, preds


def display_results(inputs, preds, super_title):

    # plot results in a grid (rows = Input/Prediction, cols = Samples)
    num_samples = len(inputs)
    fig, axs = plt.subplots(2, num_samples, figsize=(4 * num_samples, 7))

    row_titles = ['Input', 'Pred']

    for j, row_title in enumerate(row_titles):
        for i in range(num_samples):
            ax = axs[j][i]
            img = [inputs[i], preds[i]][j]
            ax.imshow(img)
            ax.axis('off')

            if j == 0:
                ax.set_title(f"Sample {i+1}", fontsize=12)

    # add row titles manually on the left of the first column
    for j, row_title in enumerate(row_titles):
        axs[j][0].text(-0.1, 0.5, row_title, va='center', ha='right', fontsize=12, transform=axs[j][0].transAxes)

    # set super title
    plt.suptitle(super_title, fontsize=16)

    plt.subplots_adjust(left=0.2, top=0.9, wspace=0.1, hspace=0.1)
    plt.show()