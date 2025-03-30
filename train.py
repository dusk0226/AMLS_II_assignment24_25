""" Use this file to train and sava the model. """
import tensorflow_datasets as tfds
from scr.data_processing import extract_cropped_data, extract_data, TFToTorchDataset
from scr.vdsr_model import VDSR, train_VDSR, weights_init
import os
import torch

""" Global variables """
# Variables to load the data:
div2k_subset = "div2k/unknown_x3" # The link of div2k subset in tensorflow dataset.

# Variables to process the data:
ds_factor = 3 # The down scale factor of this dataset.
patch_size=(64,64) # The patch size to crop the image.

# Training hyper-parameters:
batch_size = 32 
epochs = 10
learning_rate = 1e-4
num_layers = 18 # Number of convolutional layers.
channels = 64 # The number of channels (filters) of convolutional layers.
input_channels = 3 # The input and output channel number (1 for grayscale and 3 for colored image). 
residual_rate = 0.95 # How much to preserve input value for residual learning

model_load = False # If train based on a pretrained model.
model_save = False # If save the trained model.

weights_path = "./track2/vdsr_weights_track2.pth"

def main():
    """ Load and process the data. """
    # Load div2k data from tensorflow dataset.
    dataset = tfds.load(div2k_subset, data_dir="./dataset")

    # Crop the images in training dataset for faster training.
    dataset_train = extract_cropped_data(
        dataset['train'].take(800), patch_size, ds_factor)

    # Convert TensorFlow dataset to PyTorch dataset.

    dataset_train = TFToTorchDataset(dataset_train)

    """ Create or load a model. Train and save it. """
    # Initialize the model.
    VDSR_model = VDSR(channels, input_channels, 
                      num_layers, ds_factor, residual_rate)

    if os.path.exists(weights_path):
        print("Existing model weights loaded.")
        VDSR_model.load_state_dict(torch.load(weights_path))
    else:
        print("No previous weights found. Training from scratch!")
        VDSR_model.apply(weights_init)

    # Train the model.
    VDSR_model = train_VDSR(VDSR_model, epochs, 
                            dataset_train, batch_size, learning_rate)
        
    torch.save(VDSR_model.state_dict(), weights_path)

if __name__ == "__main__":
    main()