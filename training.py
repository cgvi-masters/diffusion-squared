import torch
import matplotlib.pyplot as plt
from tqdm import tqdm   # prints progress of training
import numpy as np
import pandas as pd
import os

def sample_batch(data_loader):
    # list of len=num outputs of getitem (so 2 here)
    # one list has all the anatomical images, one has the diffusion images
    batch = next(iter(data_loader)) 
    
    # get bval and bvec
    bval = batch[2][0][0] * 5000    # un-normalize for display
    bvec = batch[2][0][1:4].sqrt()  # undo transformation

    # requires formatting otherwise it prints the full precision of the 32bit float
    print("bval: {:.4f}".format(bval.item()))  
    print("bvec:", [round(x, 4) for x in bvec.tolist()])

    # get first anatomical img and corresponding diffusion image
    anat = batch[0][0,0,:]
    dw = batch[1][0,0,:]

    fig, axs = plt.subplots(1, 2, figsize=(5, 10))
    axs[0].imshow(anat, cmap='grey')
    axs[0].set_title("Anatomical Image")
    axs[1].imshow(dw, cmap='grey')
    axs[1].set_title("Diffusion Image")
    plt.tight_layout()
    plt.show()


def train_diffusion_model(model, device, train_set, train_loader, loss_fn, optimizer,
                epochs, batch_size, learning_rate, timesteps, beta_start, beta_end):
    
    # save img data dimensions
    img_shape = next(iter(train_loader))[0][0].shape

    # create linear noise scheduler based on DDPM
    betas, alphas, alphas_bar = get_noise_scheduler('linear', timesteps, beta_start, beta_end, device)

    # ensure dir exists to save model
    os.makedirs('models_diffusion', exist_ok=True)

    # cur model numnber
    model_num = len([f for f in os.listdir('models_diffusion')]) + 1
    model_path = f"models_diffusion/model{model_num}.pth"

    # pandas df to store train/test loss for plotting
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss'])

    # best loss initialized at inf
    best_loss = np.inf

    # iteratively train and test
    for epoch in tqdm(range(epochs)):

        train_loss = training_loop(model, train_loader, loss_fn, optimizer, device, timesteps, alphas_bar)

        # add losses to log
        loss_df.loc[epoch] = [epoch + 1, train_loss]

        # save model with lowest test loss (also store meta data)
        if train_loss < best_loss:
            torch.save({
                'epoch': epoch + 1,
                'batch_size': batch_size,
                'num_samples': len(train_set),
                'learning_rate': learning_rate,
                'model_state_dict': model.state_dict(),
                }, model_path)
        
        if epoch % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.5f}")

        # visualize evolution of denoising
        #if epoch % 50 == 0:
        #    model.eval()
        #    with torch.no_grad():
        #        sample(model, 1, timesteps, betas, alphas, alphas_bar, img_shape, device)
        #    model.train()
            
    # display loss curve
    plot_loss_curve(loss_df)


# creates the noise scheduler (can chooose type: linear, cosine, etc.)
def get_noise_scheduler(sched_type, timesteps, beta_start, beta_end, device):
    if sched_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)      # noise added relative to previous timestep
    elif sched_type == 'cosine':
        pass

    # closed form solution for how much noise is added at each step given t and x0
    alphas = 1. - betas                                # how much of the signal remains after timestep t
    alphas_bar = torch.cumprod(alphas, dim=0)          # how data is corrupted overtime

    return betas, alphas, alphas_bar


def training_loop(model, dataloader, loss_fn, optimizer, device, timesteps, alphas_bar):

    model.train()
    train_loss = 0

    # iterate through dataloader by batch. # anat_img_slice, dw_img_slice, (bval, bvec)
    for input, target, acq_param in dataloader:

        # send data to device
        input, target, acq_param = input.to(device), target.to(device), acq_param.to(device)

        # decide if training on blobs or shadows
        #if signal_type == 'blobs': signal = input 
        #elif signal_type == 'shadows': signal = target
        signal = target

        # sample timesteps for each image in batch
        t = torch.randint(0, timesteps, (signal.shape[0],)).to(device)

        # noise the data
        noise = torch.randn_like(signal)      # generate gaussian noise: each pixel sampled from N(0,1), values mostly between -3 and 3
        noisy_signal = noise_data(signal, t, noise, alphas_bar)

        # create conditioning vector by concatenating timestep and acquisition parameters ([B, 2])
        t_scaled = t.unsqueeze(1) / timesteps   # rescale and reshape to [B, 1]
        cond_vec = torch.cat([t_scaled, acq_param], dim=1)  # new shape [B, 3]

        # condition on blob shape image too (by concatenating to noisy image)
        combo_signal = torch.cat([noisy_signal, input], dim=1)

        # forward pass to recover noise
        pred = model(combo_signal, cond_vec)

        # calculate loss between predicted and true noise
        loss = loss_fn(pred, noise)
        train_loss += loss.item()

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)  # average loss per batch


# noise image to specified noise level
def noise_data(x, t, noise, alphas_bar):
    # add extra dims for broadcasting
    alphas_bar_t = alphas_bar[t][..., None, None, None]

    # scale img and noise accordingly
    noisy_img = alphas_bar_t.sqrt() * x + (1. - alphas_bar_t).sqrt() * noise
    return noisy_img


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