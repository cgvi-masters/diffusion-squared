import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from .database_mri import MRImagesDB
from torch.utils.data import DataLoader
from .architecture_unet import UNet

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


''' Assorted helper functions'''


# set the device depending on available GPU
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0") 
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    return device


# dynamically set number of workers to optimize use of cores
def get_num_workers():
    # unstable on mac, faster with no workers
    if torch.backends.mps.is_available():
        print("Number of workers:", 0)
        return 0
    
    try:
        num_cpus = os.cpu_count()
        # heuristic: leave 1–2 cores free
        workers = max(1, num_cpus - 2)
    except:
        workers = 4  # fallback default
    
    print("Number of workers:", workers)

    return workers


def load_model(model_path, device, cond_dim, in_chan=1, out_chan=1):

    # load saved model weights (only saved model weights)
    model_dict = torch.load(model_path, map_location=device)

    # create model and set to eval mode
    loaded_model = UNet(cond_dim=cond_dim, in_chan=in_chan, out_chan=out_chan)
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


# helper for making rotating bvec GIF
def plot_bvec(bvec, ax):
    
    ax.plot([-1,1,1,-1,-1],[1,1,-1,-1,1],[-1,-1,-1,-1,-1], color=[0.9]*3, zorder=1)
    ax.plot([-1,1,1,-1,-1],[1,1,1,1,1],[1,1,-1,-1,1], color=[0.9]*3, zorder=1)
    ax.plot([-1,-1,-1,-1,-1],[-1,1,1,-1,-1],[1,1,-1,-1,1], color=[0.9]*3, zorder=1)
    
    # ax.quiver(0, 0, 0, bvec[0], bvec[1], bvec[2], color='red', arrow_length_ratio=0.3)
    ax.plot([-bvec[0],bvec[0]],[-bvec[1],bvec[1]],[-bvec[2],bvec[2]], color=[0.5]*3)
    ax.plot(-bvec[0],-bvec[1],-bvec[2], '.', color='r', markersize=10)
    ax.plot(bvec[0],bvec[1],bvec[2], '.', color='b', markersize=10)
    
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, linewidth=0)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    
    return ax


# resolve these imports later
from src_diffusion.training import get_noise_scheduler
from src_diffusion.testing import sample
from src_unet.testing import test_unet

def create_rotation_gif_frames(method, subject, slice_idx, data_dir_path, save_dir, loaded_model, device):
    # given an anatomical image at a specific subject / slice (not random)
    # generate the GIF with rotating bvec and resulting dw images at different bvalues

    # create save dir if doesnt exits
    os.makedirs(save_dir, exist_ok=True)

    # use test data
    img_dir_path = os.path.join(data_dir_path, 'test')

    # get unique bvals [0, 650, 1000, 2000]
    bvals_path = data_dir_path + 'bvals_round.bval'
    bvals = np.unique(np.loadtxt(bvals_path)).astype(np.int16)

    # hardcode bvec path (not needed) and volume dims
    bvecs_path = data_dir_path + 'bvecs.bvec'
    volume_dims = [1, 96, 96, 70]

    # sample at 100 different bvecs
    n = 100
    bvecs = np.zeros((3,0))
   
    for i in range(n+1):
        theta = i * 4*np.pi / n
        z = 2 * i / n - 1
        r = np.sqrt(1 - z**2)
        

        fig = plt.figure(figsize=(14, 4))  # wider
        #fig.suptitle("U-Net Q-space Sampling", fontsize=16, y=0.95)

        axs = [fig.add_subplot(1, len(bvals)+1, 1)]  # static image (left-most, no 3D)
        axs += [fig.add_subplot(1, len(bvals)+1, 2, projection='3d')]  # bvec 3D plot
        axs += [fig.add_subplot(1, len(bvals)+1, b+3) for b in range(len(bvals)-1)]  # DWIs

        #fig = plt.figure(figsize=(12, 4))
        #axs = [fig.add_subplot(1, len(bvals), 1, projection='3d')]
        #axs += [fig.add_subplot(1, len(bvals), b+2) for b in range(len(bvals)-1)]
        
        bvec = np.array([r*np.cos(theta), r*np.sin(theta), z])
        bvecs = np.concatenate((bvecs,bvec[...,None]), axis=1)
        axs[1].plot(bvecs[0,:],bvecs[1,:],bvecs[2,:], color='b')
        plot_bvec(bvec, axs[1])
        axs[1].set_title('b-vector')

        
        # go through non-zero bvals [650, 1000, 2000]
        for b in range(1, len(bvals)):
            
            # number of samples
            num_samples = 3

            # create dataloader with specific bval/bvec/slice/subject
            # slice axis set = 2 as these models were only trained on horizontal slices
            bval = bvals[b]
            test_set = MRImagesDB(img_dir_path, bvals_path, bvecs_path, volume_dims, num_samples=num_samples, slice_axis=2, 
                        subject=subject, slice_idx=slice_idx, bval=bval, bvec=bvec)
            test_loader = DataLoader(dataset=test_set, batch_size=num_samples)

            
            # determine method: diffusion, latent diffusion, or unet
            if method == "diffusion":
                sampler = 'ddpm'
                timesteps = 1000
                beta_start, beta_end = 1e-4, 0.02
                img_shape = volume_dims[:3]
                
                #betas, alphas, alphas_bar = get_noise_scheduler('linear', timesteps, beta_start, beta_end, device)
                ddpm_sample, x, blobs = sample(loaded_model, test_loader, num_samples, timesteps, beta_start, beta_end, img_shape, device, sampler=sampler)
                yhat = ddpm_sample.detach().cpu().numpy()[0][0,:,:]
                anat_img = blobs.detach().cpu().numpy()[0][0,:,:]

            elif method == "unet":
                inputs, targets, preds = test_unet(loaded_model, test_set, device)
                yhat = preds[0]
                anat_img = inputs[0]

            # rotate yhat before plotting
            yhat = np.rot90(yhat, k=3)  # rotate 90
            anat_img = np.rot90(anat_img, k=3)
                
            axs[b+1].imshow(yhat, origin='lower', cmap='gray', vmin=0, vmax=1)
            axs[b+1].axis('off'); axs[b+1].set_title('b='+str(bvals[b])); 
        
        # plot anat image in front
        axs[0].imshow(anat_img, cmap="gray", origin="lower")
        axs[0].set_title("T1w")
        axs[0].axis("off")

        # save frame
        plt.savefig(os.path.join(save_dir, f"frame_{i:03d}.png"))
        #plt.show()
        plt.close(fig)


def create_rotation_gif(frame_dir, gif_name):

    # load frames in sorted order
    frames = []
    for fname in sorted(os.listdir(frame_dir)):
        if fname.endswith(".png"):
            frames.append(imageio.imread(os.path.join(frame_dir, fname)))

    # save as GIF
    imageio.mimsave(gif_name, frames, fps=10, loop=0)  # fps controls speed



# compute SSIM and PSNR metrics
def batch_metrics(preds, targets):
    
    ssim_vals, psnr_vals = [], []
    
    for i in range(len(preds)):
        pred = preds[i].astype(np.float64)
        target = targets[i].astype(np.float64)

        data_range = target.max() - target.min()
        
        # compute ssim
        ssim_val = ssim(pred, target, data_range=data_range)
        ssim_vals.append(ssim_val)

        # compute psnr
        psnr_val = psnr(pred, target, data_range=data_range)
        psnr_vals.append(psnr_val)

    return ssim_vals, psnr_vals