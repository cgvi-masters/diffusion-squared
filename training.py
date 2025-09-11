import torch
import matplotlib.pyplot as plt
from tqdm import tqdm   # prints progress of training
import numpy as np
import pandas as pd
import os

def training_loop(model, dataloader, loss_fn, optimizer, device):

    model.train()
    train_loss = 0

    # iterate through dataloader by batch
    for input, target, source in dataloader:

        # send data to device
        input, target, source = input.to(device), target.to(device), source.to(device)

        # forward pass
        pred = model(input, source)

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
        for input, target, source in dataloader:

            # send data to device
            input, target, source = input.to(device), target.to(device), source.to(device)

            # forward pass
            pred = model(input, source)

            # calculate loss
            loss = loss_fn(pred, target)
            test_loss += loss.item()

    return test_loss / len(dataloader)  # average loss per batch


def sample_batch(data_loader):

    batch = next(iter(data_loader))

    fig, axs = plt.subplots(1, 2, figsize=(5, 10))
    axs[0].imshow(batch[0][0, 0, :])  
    axs[1].imshow(batch[1][0, 0, :])  
    plt.tight_layout()
    plt.show()


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


def train_model(model, device, train_loader, val_loader, loss_fn, optimizer, 
                epochs, batch_size, learning_rate, train_set):

    # cur model numnber
    model_num = len([f for f in os.listdir('models')]) + 1
    model_path = f"models/shadow_maker{model_num}.pth"

    # pandas df to store train/test loss for plotting
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])

    # iteratively train and test
    for epoch in tqdm(range(epochs)):

        # best loss initialized at inf
        best_loss = np.inf

        train_loss = training_loop(model, train_loader, loss_fn, optimizer, device)
        test_loss = testing_loop(model, val_loader, loss_fn, device)

        # add losses to log
        loss_df.loc[epoch] = [epoch + 1, train_loss, test_loss]

        # save model with lowest test loss (also store meta data)
        if test_loss < best_loss:
            torch.save({
                'epoch': epoch + 1,
                'batch_size': batch_size,
                'num_samples': len(train_set),
                'learning_rate': learning_rate,
                'model_state_dict': model.state_dict(),
                }, model_path)
        
        print(f"  Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.4f} - Test loss: {test_loss:.4f}")

    # display loss curve
    plot_loss_curve(loss_df)