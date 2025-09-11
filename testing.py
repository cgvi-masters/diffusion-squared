import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .architecture import UNet
from src.database import LightSourceDB


def load_unet(model_path, device, in_chan, out_chan, cond_dim):

    # load saved model weights (only saved model weights)
    model_dict = torch.load(model_path, map_location=device)

    # create model and set to eval mode
    loaded_model = UNet(in_chan=in_chan, out_chan=out_chan, cond_dim=cond_dim)
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


def denoise_data(samples, i, pred, z, betas, alphas, alphas_bar):
    # get the scheduler values for the current timestep
    b_t = betas[i]      # b_t = betas[i-1]
    a_t = alphas[i]
    ab_t = alphas_bar[i]

    noise = b_t.sqrt() * z
    mean = (samples - (b_t * pred / (1 - ab_t).sqrt())) / a_t.sqrt()
    return mean + noise


from src_diffusion.training import get_noise_scheduler

def sample_latent(loaded_model, data_loader, n_samples, timesteps, beta_start, beta_end, device,
           sampler, encoder_blob, decoder_shadow, ddim_steps=None, ddim_eta=0.0):

    # generate noise schedule
    betas, alphas, alphas_bar = get_noise_scheduler('linear', timesteps, beta_start, beta_end, device)

    # generate samples from dataset for conditioning
    batch = next(iter(data_loader))
    blobs = batch[0].to(device)             # extract the blob images
    gt_shadows = batch[1].to(device)        # extract the ground truth shadows
    acq_param = batch[2].to(device)         # extract the acq params

    # NEWWWWWWW
    # embed blob images with encoder
    latent_blobs = encoder_blob(blobs)

    # generate noise samples    (must match shape of latent space)
    latent_dims = [8, 24, 24]       # HARDCODED  [8, 16, 16]
    samples = torch.randn(n_samples, latent_dims[0], latent_dims[1], latent_dims[2]).to(device)
    #samples = torch.randn(n_samples, 1, img_shape[1], img_shape[2]).to(device)
    
    # save snapshots during generation
    snapshots = []
    save_interval = 100

    # precompute ddim schedule if needed
    if sampler == "ddim":
        if ddim_steps is None:
            ddim_steps = timesteps
        step_ratio = timesteps // ddim_steps
        ddim_timesteps = np.asarray(list(range(0, timesteps, step_ratio)))
        alphas_bar_ddim = alphas_bar[ddim_timesteps]
    else:
        ddim_timesteps = None

    with torch.no_grad():

        if sampler == 'ddpm':
            for i in reversed(range(timesteps)):  # range goes from t-1 to 0
                t = torch.full((n_samples, 1), i / timesteps, device=device)  # reshape timestep tensor and rescale
                cond_vec = torch.cat([t, acq_param], dim=1)  # concat t and acq_param
                z = torch.randn_like(samples) if i > 0 else torch.zeros_like(samples)    # sample random noise to add back in
                combo_signal = torch.cat([samples, latent_blobs], dim=1)  # concatenate noisy shadow image with blob image

                # forward pass through model to predict noise
                pred = loaded_model(combo_signal, cond_vec)         

                # denoise
                samples = denoise_data(samples, i, pred, z, betas, alphas, alphas_bar)

                if i % save_interval == 0:

                    # NEWWWWW
                    # decode the predictions using the shadow decoder
                    decoded_samples = decoder_shadow(samples)
                    snap = decoded_samples.detach().squeeze(dim=1).cpu().numpy()
                    snapshots.append(snap)

        elif sampler == 'ddim':
            for idx in reversed(range(len(ddim_timesteps))):
                i = ddim_timesteps[idx]
                t = torch.full((n_samples, 1), i / timesteps, device=device)
                cond_vec = torch.cat([t, acq_param], dim=1)  # concat t and acq_param
                combo_signal = torch.cat([samples, latent_blobs], dim=1)  # concatenate noisy shadow image with blob image

                pred_noise = loaded_model(combo_signal, cond_vec)

                alpha = alphas[i]
                alpha_bar = alphas_bar[i]
                x0_pred = (samples - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()

                if idx > 0:
                    i_prev = ddim_timesteps[idx - 1]
                    alpha_bar_prev = alphas_bar[i_prev]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                sigma = ddim_eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
                noise = sigma * torch.randn_like(samples) if idx > 0 else 0

                samples = alpha_bar_prev.sqrt() * x0_pred + ((1 - alpha_bar_prev - sigma**2).sqrt()) * pred_noise + noise

                if idx % max(1, len(ddim_timesteps) // 10) == 0 or idx == 0:
                    snap = samples.detach().squeeze(dim=1).cpu().numpy()
                    snapshots.append(snap)


    
    # Plot: each row = sample, each col = timestep snapshot
    num_cols = len(snapshots) + 2
    fig, axs = plt.subplots(n_samples, num_cols, figsize=(3 * num_cols, 3 * n_samples))

    if n_samples == 1:
        axs = np.expand_dims(axs, 0)  # make it 2D array for consistency

    for col_idx, imgs in enumerate(snapshots):
        for row_idx in range(n_samples):
            axs[row_idx, col_idx].imshow(imgs[row_idx], cmap='gray', vmin=0, vmax=1)
            axs[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axs[row_idx, col_idx].set_title(f"Timestep {col_idx}")

            
    # plot ground truth shadow images in second to last column
    for row_idx in range(n_samples):
        axs[row_idx, -2].imshow(gt_shadows[row_idx].detach().cpu().numpy()[0], cmap='gray', vmin=0, vmax=1)
        axs[row_idx, -2].axis("off")
        if row_idx == 0:
            axs[row_idx, -2].set_title("Ground Truth")
    
    # plot blob images in last column
    for row_idx in range(n_samples):
        axs[row_idx, -1].imshow(blobs[row_idx].detach().cpu().numpy()[0], cmap='gray', vmin=0, vmax=1)
        axs[row_idx, -1].axis("off")
        if row_idx == 0:
            axs[row_idx, -1].set_title("Shape Image")

        # plot star for light source location (last column only)
        #source = acq_param[row_idx].detach().cpu().numpy()
        #source = source * 31   # scale by radius 
        #source = np.round(source + (img_shape[1]-1)/2).astype(int)  # shift from central coord to pixel coords
        #axs[row_idx, -1].plot(*np.flip(source), '*', markersize=20, color='white', markeredgecolor='black', markeredgewidth=1)

    plt.tight_layout()
    plt.show()
    


    # return the decoded samples and the ground truth dmri images
    return decoded_samples, gt_shadows, blobs
