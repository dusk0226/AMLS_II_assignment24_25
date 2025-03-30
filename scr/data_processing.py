"""This file contains code for div2k data processing."""
import tensorflow as tf
import torch

def random_crop(lr, hr, patch_size, ds_factor):
    """
    Perform random cropping on a set of lr and hr images.
    
    Args:
        lr, hr: Input low and high-resolution images.
        patch_size (tuple): Desired crop size for the low-resolution image (height, width).
        ds_factor (int): Factor by which HR is larger than LR.
    
    Returns:
        Tuple of (cropped_lr, cropped_hr), where:
        - cropped_lr has shape (patch_size[0], patch_size[1], 3)
        - cropped_hr has shape (patch_size[0] * ds_factor, patch_size[1] * ds_factor, 3)
    """
    lr = tf.image.convert_image_dtype(lr, tf.float32)
    hr = tf.image.convert_image_dtype(hr, tf.float32)
    lr_h, lr_w = tf.shape(lr)[0], tf.shape(lr)[1]

    # Ensure cropping fits within image dimensions
    lr_x = tf.random.uniform(shape=(), minval=0, maxval=lr_w - patch_size[1], dtype=tf.int32)
    lr_y = tf.random.uniform(shape=(), minval=0, maxval=lr_h - patch_size[0], dtype=tf.int32)
    
    hr_x = lr_x * ds_factor
    hr_y = lr_y * ds_factor

    # Crop the images
    cropped_lr = lr[lr_y: lr_y + patch_size[0], lr_x: lr_x + patch_size[1], :]
    cropped_hr = hr[hr_y: hr_y + patch_size[0] * ds_factor, hr_x: hr_x + patch_size[1] * ds_factor, :]

    return cropped_lr, cropped_hr

def extract_cropped_data(dataset, patch_size=(128, 128), ds_factor=2):
    """ 
    Randomly crop each pair of lr and hr images and enable eager execution. 
    Create train or validation dataset.
    
    Args: 
        dataset (tensorflow.python.data.ops.prefetch_op._PrefetchDataset): 
        The raw dataset loaded from tensorflow div2k datasets.
        patch_size (tuple): The aimed cropped size (height, width).
        ds_factor: The scale ratio of hr and lr.
        
    Returns:
        dataset_new (tensorflow.python.data.ops.map_op._MapDataset):
        The updated dataset.
    """
    dataset_new = dataset.map(lambda t: 
        (lambda cropped: {'lr': cropped[0], 'hr': cropped[1]})(random_crop(t['lr'], t['hr'], patch_size, ds_factor))
    )
    
    return dataset_new

def convert_to_tensor(image):
    formatted_image = tf.image.convert_image_dtype(image, tf.float32)
    return formatted_image

def extract_data(dataset):
    """ 
    Enable eager execution for every image. Create train or validation dataset.
    
    Args: 
        dataset (tensorflow.python.data.ops.prefetch_op._PrefetchDataset): 
        The raw dataset loaded from tensorflow div2k datasets.

    Returns:
        dataset_new (tensorflow.python.data.ops.map_op._MapDataset):
        The updated dataset.
    """
    dataset_new = dataset.map(lambda x: {'lr': convert_to_tensor(x['lr']), 'hr': convert_to_tensor(x['hr'])})
    
    return dataset_new

class TFToTorchDataset(torch.utils.data.IterableDataset):
    """ 
    Convert TensorFlow dataset to PyTorch dataset.
    (tensorflow.python.data.ops.map_op._MapDataset -> torch.utils.data.IterableDataset)
    """
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset

    def __iter__(self):
        """ 
        Convert TensorFlow tensors to NumPy, then PyTorch tensors.
        For each imgae, (H, W, C) → (C, H, W) for VDSR model.
        """
        for example in self.tf_dataset:
            lr = example["lr"].numpy()*255.0
            hr = example["hr"].numpy()*255.0
            lr = torch.from_numpy(lr).permute(2, 0, 1)  
            hr = torch.from_numpy(hr).permute(2, 0, 1)
            yield lr, hr