import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d

from .database import LightSourceDB, generate_img_pair
from .architecture import UNet

def display_results(inputs, targets, preds, super_title):

    # plot results in a grid (rows = Input/GT/Prediction, cols = Samples)
    num_samples = len(inputs)
    fig, axs = plt.subplots(3, num_samples, figsize=(4 * num_samples, 10))

    row_titles = ['Input', 'GT', 'Pred']

    for j, row_title in enumerate(row_titles):
        for i in range(num_samples):
            ax = axs[j][i]
            img = [inputs[i], targets[i], preds[i]][j]
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


def print_metadata(model_dict):

    print(f"Model Metadata:")
    print("-" * 30)
    print(f"  Epochs:          {model_dict['epoch']}")
    print(f"  Learning Rate:  {model_dict['learning_rate']}")
    print(f"  Batch Size:     {model_dict['batch_size']}")
    print(f"  Num Samples:    {model_dict['num_samples']}")


# function to compute MAE from predicted image to gt output
def compute_MAE(pred, target):

    return np.mean(np.abs(pred - target))


def plot_MAE_per_angle(num_angles, samples_per_angle, loaded_model, device):
    
    # angles
    angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False, dtype=np.float32)

    # image size and radius
    imshape = np.array([64] * 2) 
    radius = 31

    # list with angle and MAE
    MAE_list = np.zeros((2, num_angles * samples_per_angle))

    # counter 
    counter = 0

    # iterate through each degree
    for angle in angles:

        # iterate through each sample image
        for _ in range(samples_per_angle):
        
            # create input and target
            input, target, source = generate_img_pair(imshape, angle, radius)
        
            # convert to proper format for UNet model and make prediction
            pred = loaded_model(torch.from_numpy(input).unsqueeze(0).to(device),
                                torch.from_numpy(source).unsqueeze(0).to(device))
                                #torch.tensor([[angle]]).to(device))

            # convert back to plottable format
            pred = pred.detach().squeeze().cpu().numpy()

            # compute MAE and add it to list
            MAE = compute_MAE(pred, target)
            MAE_list[0, counter] = angle
            MAE_list[1, counter] = MAE

            # increment counter
            counter += 1

    # split list into angles and values
    angles_list, values_list = MAE_list

    # reshape values and take mean MAE for each angle
    values_by_angle = values_list.reshape(num_angles, samples_per_angle)
    MAE_means = values_by_angle.mean(axis=1)

    # smooth data with gaussian filter for plotting
    wrapped = np.concatenate([MAE_means[-20:], MAE_means, MAE_means[:20]])  # wraps around then smoothes
    smoothed_values = gaussian_filter1d(wrapped, sigma=20)[20:-20]

    # close the loop so ends of line meet
    closed_angles = np.append(angles, angles[0])
    closed_smoothed_values = np.append(smoothed_values, smoothed_values[0])

    # plotting
    ax = plt.subplot(projection='polar')
    ax.scatter(angles_list, values_list, color='blue', s=10, alpha=0.3, label='All Samples')  # all MAE values
    ax.plot(closed_angles, closed_smoothed_values, color='red', linewidth=2, label='filtered, sigma=3')  # curve connecting values
    plt.show()


def load_model(model_path, device):

    # load saved model weights (only saved model weights)
    model_dict = torch.load(model_path)

    # create model and set to eval mode
    loaded_model = UNet(cond_dim=2)
    loaded_model.load_state_dict(model_dict['model_state_dict'])
    loaded_model.to(device)
    loaded_model.eval()

    print_metadata(model_dict)
    
    return loaded_model


def test_model(loaded_model, device, super_title):

    # compute predictions
    test_set = LightSourceDB(num_samples=5, method="fixed", fixed_angle=np.pi)
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    inputs, targets, preds = [], [], []
    with torch.no_grad():  # disable grad during inference
        for input, target, source in test_loader:

            # send data to device and perform forward pass
            input, target, source = input.to(device), target.to(device), source.to(device)
            pred = loaded_model(input, source)

            inputs.append(input.squeeze().cpu().numpy())
            targets.append(target.squeeze().cpu().numpy())
            preds.append(pred.detach().squeeze().cpu().numpy())

    display_results(inputs, targets, preds, super_title)