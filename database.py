import numpy as np
import os

import torch
from torch.utils.data import Dataset
import torchio as tio


class MRImagesDB(Dataset):

    def __init__(self, img_dir_path, bvals_path, bvecs_path, volume_dims, num_samples, slice_axis=None):
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

        # number of samples desired (total num of images = subjects x volumes per subject x slices per volume)
        self.num_samples = num_samples

        # 2d slice axis -> default 0 = horizontal, 1 = , 2 = 
        self.slice_axis = slice_axis

    def __len__(self):
        # lets just do a subset for now (not every possible image)
        # return len(self.subdirs) * len(self.bvals) * self.volume_dims[3]
        return self.num_samples

    def __getitem__(self, idx):
        # random selections
        subject = np.random.choice(self.subdirs)  # for now only 1 subject in mini_dataset
        dw_idx = np.random.choice(self.dw_inds) 
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
        mask_slice_axis = np.any(anat_vol[0,:] > 0, axis=other_axes)                   # identify slices that are not all 0
        mask_idxs = np.where(mask_slice_axis)[0]                                  # get indices of those slices with some data
        slice_idx =  np.random.choice(range(mask_idxs[0], mask_idxs[-1]))         # random slice within non-empty range
        
        # take slices
        anat_slice = np.take(anat_vol_norm, slice_idx, axis=self.slice_axis + 1)  # +1 bc shape [1, H, W]
        dw_slice = np.take(dw_vol_norm, slice_idx, axis=self.slice_axis + 1)
        
        # get bval / bvector and combine acquisition param into single conditioning vector
        bval = self.bvals[dw_idx] / 5000  # normalize between 0-1 by dividing by largest bval most scanners can do
        bvec = self.bvecs[:, dw_idx]
        acq_param = torch.tensor([bval, *bvec], dtype=torch.float32)

        # convert to float32 tensors
        anat_slice = torch.tensor(anat_slice, dtype=torch.float32)  # shape [1, H, W]
        dw_slice = torch.tensor(dw_slice, dtype=torch.float32)
        #anat_slice = anat_slice[np.newaxis, :].astype(np.float32)
        #dw_slice = dw_slice[np.newaxis, :].astype(np.float32)
        
        return anat_slice, dw_slice, acq_param


normalize = tio.Compose([
    # transformation to normalize an image volume
    tio.Clamp(out_min=0),  # zero out negative values (like relu)
    tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))  # rescale output between 0-1 and avoid outliers
])


# function to add symmetric padding so each dimension is divisible by 2**k
def pad(volume, k):

    input_shape = np.array(volume.shape)
    target_shape = np.ceil(input_shape / 2**k) * 2**k

    pad_before = np.round((target_shape - input_shape) / 2).astype(int) # convert float to int
    pad_after = (target_shape - input_shape - pad_before).astype(int)

    padding = np.stack((pad_before, pad_after), axis=1)
    padded_vol = np.pad(volume, padding)
    
    return padded_vol