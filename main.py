""" Use this file to do the training and inference. """
import tensorflow_datasets as tfds
from scr.data_processing import extract_cropped_data, extract_data, TFToTorchDataset
from scr.vdsr_model import VDSR, train_VDSR, inference_VDSR, weights_init
from scr.envaluation import plot_images, bicubic_sr, psnr, ssim
import numpy as np
import torch

""" Global variables """
# Variables to load the data:
div2k_subset = "div2k/unknown_x3" # The link of div2k subset in tensorflow dataset.

# Variables to process the data:
ds_factor = 3 # The down scale factor of this dataset.
patch_size_train=(64,64) # The patch size to crop the image for training.
patch_size_test=(128,128) # The patch size to crop the image for testing.

# Training hyper-parameters:
batch_size = 32 
epochs = 10
learning_rate = 1e-4
num_layers = 18 # Number of convolutional layers.
channels = 64 # The number of channels (filters) of convolutional layers.
input_channels = 3 # The input and output channel number (1 for grayscale and 3 for colored image). 

# The residual_rate for inference should be the same or very close to the value used in the training. 
# The default residual_rate should be 0.95 for the saved parameters in this project.
residual_rate = 0.95 # How much to preserve input value for residual learning .

# If train the model from beginning or use trained parameters.
Train_from_beginning = False
weights_path = "./track2/vdsr_track2x3.pth" # The path where the trained parameters are saved.
pnp = False # If implement PnP-ADMM on the super-resolved results.
num_iters = 10 # The iteration times of the PnP-ADMM network.

def main():
    """ Load and process the data. """
    # Load div2k data from tensorflow dataset.
    dataset = tfds.load(div2k_subset, data_dir="./dataset")

    # Crop the images in training dataset for faster training and better comparison.
    # Use dataset_val = extract_data(dataset['validation']) if want to remain the original size.
    dataset_train = extract_cropped_data(
        dataset['train'], patch_size_train, ds_factor)

    dataset_val = extract_cropped_data(
        dataset['validation'], patch_size_test, ds_factor)

    # Convert TensorFlow dataset to PyTorch dataset.
    dataset_train = TFToTorchDataset(dataset_train)
    dataset_val = TFToTorchDataset(dataset_val)

    """ Train the model. """
    VDSR_model = VDSR(channels, input_channels, 
                      num_layers, ds_factor, residual_rate)

    if Train_from_beginning:
        # Initialize the model.
        VDSR_model.apply(weights_init)
        # Train the model.
        VDSR_model = train_VDSR(VDSR_model, epochs, 
                                dataset_train, batch_size, learning_rate)
    else:
        try:
            VDSR_model.load_state_dict(torch.load(weights_path), strict=False)
            print("Existing model weights loaded.")
        except (FileNotFoundError, RuntimeError):
            print("""No previous weights found or incompatible weights. 
                  Correct the path or train from beginning!""")

    """ Do inference. """
    lr_list, sr_list, hr_list = inference_VDSR(VDSR_model, dataset_val, pnp=pnp, num_iters=num_iters)

    """ Show and envaluate results. """
    bi_sr_list = bicubic_sr(lr_list, ds_factor)
    psnr_list = [] 
    ssim_list = []
    psnr_list_bi = []
    ssim_list_bi = []
    for sr, hr, bi_sr in zip(sr_list, hr_list, bi_sr_list):
        psnr_list.append(psnr(sr, hr))
        psnr_list_bi.append(psnr(bi_sr, hr))
        ssim_value = ssim(sr, hr)
        ssim_list.append(ssim_value)
        ssim_value_bi = ssim(bi_sr, hr)
        ssim_list_bi.append(ssim_value_bi)

    psnr_list_bi = [x for x in psnr_list_bi if x != float('inf')]
    print(psnr_list,f'\n The mean psnr of the model output is {np.mean(psnr_list)}')
    print(psnr_list_bi,f'\n The mean psnr of the bicubic output is {np.mean(psnr_list_bi)}')
    print(ssim_list,f'\n The mean ssim of the model output is {np.mean(ssim_list)}')
    print(ssim_list_bi,f'\n The mean ssim of the bicubic output is {np.mean(ssim_list_bi)}')

    p = 0
    plot_images(lr_list[p], sr_list[p], hr_list[p], './outputs/model_output_test.png')
    plot_images(lr_list[p], bi_sr_list[p] ,hr_list[p], './outputs/output_test.png')

if __name__ == "__main__":
    main()

