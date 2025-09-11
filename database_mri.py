import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import Dataset
import torchio as tio


''' Real dataset to load random anatomical / diffusion slices and aquisition settings.'''


class MRImagesDB(Dataset):

    def __init__(self, img_dir_path, bvals_path, bvecs_path, volume_dims, num_samples, 
                 slice_axis=None, subject=None, slice_idx=None, bval=None, bvec=None, dw_idx=None):
        super().__init__()
        
        # store image dir and list of subjects
        self.img_dir_path = img_dir_path
        self.subdirs = next(os.walk(img_dir_path))[1]  

        # load bvals and bvecs  (same aquisition protocol across subjects)
        self.bvals = np.loadtxt(bvals_path)   # shape (n,)
        self.bvecs = np.loadtxt(bvecs_path)   # shape (3,n)

        # indices where bvals are zero or non zero (diffusion weighted)
        self.b0_inds = np.where(self.bvals == 0)[0]
        self.dw_inds = np.where(self.bvals > 0)[0]

        # each volume has size = [1,H,W,D]  (currently 70 horizontal slices [1, 96, 96, 70])
        self.volume_dims = volume_dims

        # number of samples desired
        self.num_samples = num_samples

        # 2d slice axis -> 0 = saggital, 1 = coronal, 2 = horizontal (default)
        self.slice_axis = slice_axis

        # selected parameters if desired 
        self.subject = subject
        self.slice_idx = slice_idx
        self.bval = bval
        self.bvec = bvec
        self.dw_idx = dw_idx

    def __len__(self):
        # total num of possible images = subjects x volumes per subject x slices per volume
        return self.num_samples

    def __getitem__(self, idx):

        # random selections
        subject = np.random.choice(self.subdirs) if self.subject is None else self.subject
        # modifications for SSIM etc - use idx to get dw_inx
        #dw_idx = self.dw_inds[idx] 
        dw_idx = np.random.choice(self.dw_inds) if self.dw_idx is None else self.dw_idx
        b0_idx = np.random.choice(self.b0_inds)

        # generate pathnames
        anat_path = os.path.join(self.img_dir_path, subject, 'anat', subject + '_t1.nii.gz')
        dw_path = os.path.join(self.img_dir_path, subject, 'dwi', subject + '_dwi_preproc_' + str(dw_idx) + '.nii.gz')
        b0_path = os.path.join(self.img_dir_path, subject, 'dwi', subject + '_dwi_preproc_' + str(b0_idx) + '.nii.gz')

        # load image volumes with torchio and extract numpy arrays
        anat_vol = tio.ScalarImage(anat_path).data.numpy()     # [1, H, W, D] = [1, 96, 96, 70]
        dw_vol = tio.ScalarImage(dw_path).data.numpy()
        b0_vol = tio.ScalarImage(b0_path).data.numpy()

        # normalize
        anat_vol_norm = normalize(anat_vol)  # torchio transformation
        dw_vol_norm = np.clip(dw_vol / (b0_vol + 1e-10), 0, 1)  # normalize by b0 volume

        # create and apply mask
        mask = anat_vol > 0
        dw_vol_norm = dw_vol_norm * mask

        # pick random non-empty slice   (if slice_axis is none, keep entire 3d volume)
        other_axes = tuple(np.delete(range(3), self.slice_axis))
        mask_slice_axis = np.any(anat_vol[0,:] > 0, axis=other_axes)              # identify slices that are not all 0
        mask_idxs = np.where(mask_slice_axis)[0]                                  # get indices of non-empty slices with data and take random slice
        slice_idx = np.random.choice(range(mask_idxs[0], mask_idxs[-1])) if self.slice_idx is None else self.slice_idx       
        
        # take slices
        anat_slice = np.take(anat_vol_norm, slice_idx, axis=self.slice_axis + 1)  # +1 bc shape [1, H, W]
        dw_slice = np.take(dw_vol_norm, slice_idx, axis=self.slice_axis + 1)

        # convert np arrays to tensors
        anat_slice = torch.from_numpy(anat_slice).float()
        dw_slice = torch.from_numpy(dw_slice).float()

        # if slice is not horizontal, pad volume to square
        if self.slice_axis == 0 or self.slice_axis == 1:
            # make copy and convert to float32 tensor
            anat_slice = anat_slice.clone().detach().float() # shape [1, H, W]
            dw_slice = dw_slice.clone().detach().float()
            # pad to square
            anat_slice = pad_to_square(anat_slice)
            dw_slice = pad_to_square(dw_slice)
        
        # normalize bval between 0-1 by dividing by largest bval most scanners can do
        bval = self.bvals[dw_idx] / 5000  if self.bval is None else self.bval / 5000
        
        # transform bvec to preserve symmetry: x,y,z -> x2,y2,z2,xy,xz,yz
        bvec = self.bvecs[:, dw_idx]
        bvec_trans = transform_bvec(self.bvec if self.bvec is not None else bvec) 

        # combine acquisition param into single conditioning vector
        acq_param = torch.tensor([bval, *bvec_trans], dtype=torch.float32)      # size [1,7]
        
        return anat_slice, dw_slice, acq_param


# transform bvec from xyz to preserve symmetry
def transform_bvec(bvec):
    bvec = torch.from_numpy(bvec).float()
    x, y, z = bvec
    bvec_trans = torch.stack([x**2, y**2, z**2, x*y, x*z, y*z])
    return bvec_trans


# transformation to normalize an image volume
normalize = tio.Compose([
    tio.Clamp(out_min=0),  # zero out negative values (like relu) even though all values should be >= 0
    tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 99))  # rescale output between 0-1 and avoid outliers
])


# pads 2d slice with zeros to make it square if rectangular depending on slice dir (ex: [96,70] to [96,96])
def pad_to_square(slice_tensor, pad_value=0.0):
    C, H, W = slice_tensor.shape
    size = max(H, W)
    
    pad_h = (size - H) // 2
    pad_w = (size - W) // 2
    
    pad_top, pad_bottom = pad_h, size - H - pad_h
    pad_left, pad_right = pad_w, size - W - pad_w
    
    # F.pad expects padding in (left, right, top, bottom) order for 2D tensors      3D: [C,H,W] -> pad last two dims
    padded = torch.nn.functional.pad(slice_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)

    return padded


def sample_batch_mri(data_loader):
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

    fig, axs = plt.subplots(1, 2, figsize=(7, 12))
    axs[0].axis('off')
    axs[0].imshow(anat, cmap='grey')
    #axs[0].set_title("Anatomical Image")
    axs[1].axis('off')
    axs[1].imshow(dw, cmap='grey')
    #axs[1].set_title("Diffusion Image")
    plt.tight_layout()
    plt.show()