import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .architecture import UNet


def load_model(model_path, device, cond_dim):

    # load saved model weights (only saved model weights)
    model_dict = torch.load(model_path, map_location=device)

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


def test_unet(loaded_model, test_set, device):

    # compute predictions
    test_loader = DataLoader(dataset=test_set, batch_size=1)
    inputs, targets, preds = [], [], []

    loaded_model.eval()
    with torch.no_grad():  # disable grad during inference
        for input, target, acq_param in test_loader:

            # send data to device and perform forward pass
            input, target, acq_param = input.to(device), target.to(device), acq_param.to(device)
            pred = loaded_model(input, acq_param)

            inputs.append(input.squeeze().cpu().numpy())
            targets.append(target.squeeze().cpu().numpy())
            preds.append(pred.detach().squeeze().cpu().numpy())

    return inputs, targets, preds


def display_results(inputs, targets, preds, super_title):

    # plot results in a grid (rows = Input/GT/Prediction, cols = Samples)
    num_samples = len(inputs)
    fig, axs = plt.subplots(3, num_samples, figsize=(4 * num_samples, 10))

    row_titles = ['Input', 'GT', 'Pred']

    for j, row_title in enumerate(row_titles):
        for i in range(num_samples):
            ax = axs[j][i]
            img = [inputs[i], targets[i], preds[i]][j]
            ax.imshow(img, cmap='grey')
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