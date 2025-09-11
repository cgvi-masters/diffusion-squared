import numpy as np
import random
import matplotlib.pyplot as plt

import scipy
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol
from skimage import measure
from torch.utils.data import Dataset


''' Toy dataset to generate random shapes and corresponding shadows on the fly.'''

class LightSourceDB(Dataset):

    def __init__(self, num_samples=1000, method="random", min_angle=None, max_angle=None, fixed_angle=None):
        super().__init__()
        self.num_samples = num_samples
        self.method = method            # select method for position generation
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.fixed_angle = fixed_angle

    def __len__(self):
        # can set length as long as we want, bc generating on the fly
        return self.num_samples
    
    def __getitem__(self, idx):
        # typically called at batch level
        # no index if generating on fly

        imshape = np.array([64] * 2) # image size
        radius = 31  # fixed radius

        # create x,y coords of light source position
        angle = gen_light_pos(self.method, self.min_angle, self.max_angle, self.fixed_angle) 
        
        # original image and the resulting shadow image and source vector
        input, target, source = generate_img_pair(imshape, angle, radius)

        return (input, target, source)
    

def make_circle_image(size=64, radius=8):
    # initialize with ones
    img = np.ones((size, size), dtype=np.float32)

    # coordinates
    yy, xx = np.ogrid[:size, :size]
    center = size // 2

    # mask for circle
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2

    # set circle region to 0
    img[mask] = 0.0

    return img


def generate_img_pair(imshape, angle, radius):
    # function to generate a random shape and its shadow image

    source_vector = np.array([np.cos(angle), np.sin(angle)])  # 2d coord of angle on unit circle
    source = radius * source_vector  # scale by radius
    source = np.round(source + (imshape[0]-1)/2).astype(int) # shift from central coord to pixel coords

    
    # generate a random shape mask (catches error)
    while True:
        try:
            shape = genSegImg(imshape // 2, 5, sigma=5) > 0
            if np.any(shape):
                break
        except ValueError:
            # this catches the argmax of empty sequence issue
            print("error raised in gen seg!!")
            continue

    # create original image using shape
    input = np.ones(imshape)
    input[imshape[0]//4-1:3*imshape[0]//4-1, imshape[1]//4-1:3*imshape[1]//4-1] = ~shape
    

    input = make_circle_image(size=64, radius=8)

    # shadow computation
    target = visibility(source, input.copy())

    # convert from float64 to float32 and make 3D to account for channel (1x64x64)
    input = input[np.newaxis, :].astype(np.float32)
    target = target[np.newaxis, :].astype(np.float32)
    source_vector = source_vector.astype(np.float32)

    # invert images (so background is 0 instead of 1 to account for zero padding)
    input, target = 1 - input, 1 - target

    
    # make a circular mask
    Y, X = np.ogrid[:64, :64]
    center = (64 // 2, 64 // 2)
    radius = min(64, 64) // 2  # inscribed circle
    dist = (X - center[1])**2 + (Y - center[0])**2
    mask = dist <= radius**2

    # apply mask to both images before MAE to avoid corner effects
    input = input * mask
    target = target * mask
    

    # return the original image, the resulting shadow image, and the angle coords
    return (input, target, source_vector)
    


def gen_light_pos(method, min_angle, max_angle, fixed_angle):

    if method=="fixed":
        angle = fixed_angle

    elif method=="corners":  # pos is one of 4 corners
        corner_angles = [0, np.pi / 2, np.pi, -np.pi / 2]   # in radians
        rand_angle = random.choice([0, 1, 2, 3])
        angle = corner_angles[rand_angle]

    elif method=="random":
        angle = np.pi * (2 * np.random.rand() - 1) # radians between -pi and pi

    elif method=="constrained":  # constrains pos between min and max angle (radians from -pi to pi)
        angle = min_angle + (max_angle - min_angle) * np.random.rand()

    return angle


def sample_batch_toy(data_loader):

    batch = next(iter(data_loader))

    fig, axs = plt.subplots(1, 2, figsize=(10, 14))
    axs[0].imshow(batch[0][0, 0, :]) 
    #axs[0].set_title("Shape Image") 
    axs[0].axis('off')
    axs[1].imshow(batch[1][0, 0, :])  
    #axs[1].set_title("Shadow Image")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    print(batch[2][0])


def plot_random_pair():
    # image size (64x64)
    imshape = np.array([64] * 2)

    # calculate location of light source given angle/radius
    #np.random.seed(54321)  # fixed direction
    angle = np.pi * (2*np.random.rand()-1)
    radius = 31
    source = radius * np.array([np.cos(angle), np.sin(angle)])
    source = np.round(source + (imshape[0]-1)/2).astype(int) # shift from central coord to pixel coords

    # generate random shape and its border
    shape = genSegImg(imshape//2, 5, sigma=5) > 0
    contours = measure.find_contours(shape, fully_connected='high')

    # plot figure
    plt.figure(dpi=300)

    lum = np.ones(imshape)
    lum[imshape[0]//4-1:3*imshape[0]//4-1, imshape[1]//4-1:3*imshape[1]//4-1] = ~shape
    #lum = make_circle_image(size=64, radius=8)

    contours = [contour + np.array([imshape[0]/4-1,imshape[1]/4-1]) for contour in contours]

    # defines center and number of dots for blue circle
    center = (imshape-1)/2
    theta = np.linspace(0, 2*np.pi, 100)  # 100 points from 0 to 2π

    plt.subplot(1,2,1)
    plt.imshow(lum, cmap='gray', vmax=1.3, origin='lower')
    plt.plot(radius*np.cos(theta)+center[0], radius*np.sin(theta)+center[1], ':', color='b')  # dotted circle
    plt.plot(*np.flip(source), '*', markersize=20)  # star
    
    plt.axis('off')

    # shadow computation
    lum = visibility(source, lum)

    lum = 1 - lum #, target = 1 - input, 1 - target

    plt.subplot(1,2,2)
    plt.imshow(lum, origin='lower') #, vmin=0, vmax=1.3, origin='lower')  # cmap='hot',
    plt.axis('off')

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=[1]*3)  # border around shape
            
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------------

# Antoine's code to generate shadow images


def visibility_from_corner(grid):
    
    for x in range(grid.shape[0]):
        for y in range(int(x==0), grid.shape[1]):
            grid[x,y] *= (x*grid[x-1,y] + y*grid[x,y-1]) / (x + y)
            
    return

def visibility(source, grid):

    #grid = grid1.copy() # copy so doesn't affect original

    visibility_from_corner(grid[source[0]:,source[1]:])
    visibility_from_corner(grid[source[0]::-1,source[1]:])
    visibility_from_corner(grid[source[0]::-1,source[1]::-1])
    visibility_from_corner(grid[source[0]:,source[1]::-1])
    
    return grid

# changed npts to 128 so power of 2 to supress warning, makes shapes more pointy/detailed
def genSegImg(volshape, nlabs, npts=64, sigma=25, scal=0.8, seed=None,
              single_cc=True, rm_border=True):
    
    volshape = np.array(volshape)
    ndims = len(volshape)
    grid = np.indices(volshape).reshape(ndims, -1).T
    
    seg = np.zeros((grid.shape[0], nlabs+1))
    sobol = Sobol(ndims, seed=seed)
    for j in range(nlabs+1):
        pts = (scal*(sobol.random(npts)-0.5) + 0.5)*(volshape-1)
        
        sqdist = cdist(grid, pts, 'sqeuclidean')
        wmap = np.exp(-sqdist / sigma**2)
        seg[:,j] = np.sum(wmap, axis=1)
             
    seg = np.argmax(seg, axis=-1).reshape(volshape)
  
    if rm_border:

        # relabel
        seg2 = np.zeros_like(seg)
        currlab = 0
        for lab in np.unique(seg):
            if lab == 0:  
                continue
            
            mask = (seg == lab)
            labreg, nbits = scipy.ndimage.label(mask)
            labreg[labreg > 0] += currlab
            seg2 += labreg

            currlab += nbits

        # remove border regions
        if ndims == 2:
            edges = np.concatenate([seg2[0, :], seg2[-1, :], 
                                    seg2[:, 0], seg2[:, -1]])
        elif ndims == 3:
            edges = np.concatenate([seg2[0, :, :].ravel(),seg2[-1, :, :].ravel(), 
                                    seg2[:, 0, :].ravel(),seg2[:, -1, :].ravel(),
                                    seg2[:, :, 0].ravel(),seg2[:, :, -1].ravel()])
        edge_labels = np.unique(edges)
        mask = np.isin(seg2, edge_labels)
        seg2[mask] = 0
        seg = np.where(seg2 > 0, seg, 0)
    
    if single_cc:
        # only keep the biggest connected component
        seg0 = (seg > 0).astype(np.int16)
        chunks = measure.label(seg0)
        chunk_sizes = np.bincount(chunks.ravel())[1:]
        seg = seg * (chunks == np.argmax(chunk_sizes)+1)

    return seg