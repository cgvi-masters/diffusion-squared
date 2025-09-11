import torch
import matplotlib.pyplot as plt
from tqdm import tqdm   # prints progress of training
import numpy as np
import pandas as pd
import os


''' ----------------------- Training for U-Net -----------------------'''


def training_loop(model, dataloader, loss_fn, optimizer, device):

    model.train()
    train_loss = 0

    # iterate through dataloader by batch
    for input, target, acq_param in dataloader:

        # send data to device
        input, target, acq_param = input.to(device), target.to(device), acq_param.to(device)

        # forward pass
        pred = model(input, acq_param)

        # calculate loss
        loss = loss_fn(pred, target)
        train_loss += loss.item()

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)  # average loss per batch


def testing_loop(model, dataloader, loss_fn, device):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for input, target, acq_param in dataloader:

            # send data to device
            input, target, acq_param = input.to(device), target.to(device), acq_param.to(device)

            # forward pass
            pred = model(input, acq_param)

            # calculate loss
            loss = loss_fn(pred, target)
            test_loss += loss.item()

    return test_loss / len(dataloader)  # average loss per batch


def train_unet(model, device, train_loader, val_loader, loss_fn, optimizer, 
                epochs, batch_size, learning_rate, num_samples):

    # cur model numnber
    model_num = len([f for f in os.listdir('models')]) + 1
    model_path = f"models/unet_{model_num}.pth"

    # pandas df to store train/test loss for plotting
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])

    # best loss initialized at inf
    best_loss = np.inf

    # iteratively train and test
    for epoch in tqdm(range(epochs)):

        train_loss = training_loop(model, train_loader, loss_fn, optimizer, device)
        val_loss = testing_loop(model, val_loader, loss_fn, device)

        # add losses to log
        loss_df.loc[epoch] = [epoch + 1, train_loss, val_loss]

        # save model with lowest test loss (also store meta data)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'batch_size': batch_size,
                'num_samples': num_samples,
                'learning_rate': learning_rate,
                'model_state_dict': model.state_dict(),
                }, model_path)
        
        print(f"  Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

    # display loss curve
    plot_loss_curve(loss_df)


def plot_loss_curve(loss_df):

    plt.figure(figsize=(8, 5))

    plt.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')
    plt.plot(loss_df['epoch'], loss_df['test_loss'], label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.title('Training and Testing Loss per Epoch')
    plt.yscale('log')  # plotted on log scale to enhance small diff
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()



''' ----------------------- Testing for U-Net -----------------------'''

from torch.utils.data import DataLoader
from .architecture_unet import UNet


def load_model(model_path, device, cond_dim):

    # load saved model weights (only saved model weights)
    model_dict = torch.load(model_path)

    # create model and set to eval mode
    loaded_model = UNet(cond_dim=cond_dim)
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


def test_unet(loaded_model, test_set, device, super_title):

    # compute and store predictions
    test_loader = DataLoader(dataset=test_set, batch_size=1)
    inputs, targets, preds = [], [], []

    # model in eval mode and disable grad during inference
    loaded_model.eval()
    with torch.no_grad():  
        for input, target, acq_param in test_loader:

            # send data to device and perform forward pass
            input, target, acq_param = input.to(device), target.to(device), acq_param.to(device)
            pred = loaded_model(input, acq_param)

            inputs.append(input.squeeze().cpu().numpy())
            targets.append(target.squeeze().cpu().numpy())
            preds.append(pred.detach().squeeze().cpu().numpy())

    display_results(inputs, targets, preds, super_title)

    print(compute_MAE(preds[0], targets[0]))


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



''' ----------------------- MAE Interpolation Experiment -----------------------'''


from scipy.ndimage import gaussian_filter1d
from .database_toy import genSegImg, visibility

# function to compute MAE from predicted image to gt output
def compute_MAE(pred, target):
    '''
    # get imshape
    H, W = pred.shape  # [64, 64]
    
    # strip channel from target [1, 64, 64] -> [64, 64]
    target = target[0]
    
    
    # make a circular mask
    Y, X = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    radius = min(H, W) // 2  # inscribed circle
    dist = (X - center[1])**2 + (Y - center[0])**2
    mask = dist <= radius**2

    # apply mask to both images before MAE to avoid corner effects
    masked_pred = pred * mask
    masked_target = target * mask
    

    MAE = np.mean(np.abs(masked_pred - masked_target)) / np.sum(mask)
    '''
    MAE = np.mean(np.abs(pred - target))

    return MAE


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

    max_MAE = -np.inf
    max_MAE_angle = 0
    big_MAE_angle_list = []

    loaded_model.eval()
    with torch.no_grad(): 

        # iterate through each degree
        for angle in angles:

            # iterate through each sample image
            for _ in range(samples_per_angle):
            
                # create input and target
                input, target, source = generate_img_pair(imshape, angle, radius)
            
                # convert to proper format for UNet model 
                input_tensor = torch.from_numpy(input).unsqueeze(0).to(device)
                source_tensor = torch.from_numpy(source).unsqueeze(0).to(device)

                #plt.imshow(input[0])
                #plt.show()

                # make prediction
                pred = loaded_model(input_tensor, source_tensor)

                # convert back to plottable format
                pred = pred.detach().squeeze().cpu().numpy()

                #plt.imshow(target[0])
                #plt.show()
                #plt.imshow(pred)
                #plt.show()

                # get rid of extra dim from target
                target = target[0]
                # create mask where values are non-zero in target
                mask = target != 0
                # apply mask to get non zero values in pred and target
                non_zero_pred = pred[mask]
                non_zero_target = target[mask]
                print(non_zero_pred.shape)
                MAE = compute_MAE(non_zero_pred, non_zero_target)

                # compute MAE and add it to list
                #MAE = compute_MAE(pred, target)
                MAE_list[0, counter] = angle
                MAE_list[1, counter] = MAE

                if MAE > max_MAE:
                    max_MAE = MAE
                    max_MAE_angle = angle

                if MAE > 0.008:
                    big_MAE_angle_list.append(angle)

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

    print(max_MAE_angle)
    print(big_MAE_angle_list)



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

    '''
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
    '''

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