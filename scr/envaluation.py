""" This file contains the functions helping evaluate the model performance
    (The quality of super-resolved images). """
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity

def plot_images(lr, sr, hr, name):
    fig, axe = plt.subplots(1,3)
    axe[0].imshow(lr)
    axe[0].set_title("Low-Resolution")
    axe[1].imshow(sr)
    axe[1].set_title("Super-Resolved")
    axe[2].imshow(hr)
    axe[2].set_title("High-Resolution")

    for ax in axe:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches="tight", pad_inches=0)
    return fig, axe

def bicubic_sr(images, ds_factor):
    sr_images = []
    for image in images:
        new_h, new_w = image.shape[0]*ds_factor, image.shape[1]*ds_factor
        sr_image  = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        sr_images.append(sr_image)
 
    return sr_images

def psnr(sr, hr):
    mse = np.mean((sr-hr)**2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20*math.log10(255.0/math.sqrt(mse))

    return psnr

def ssim(sr, hr):
    ssim = structural_similarity(sr, hr, channel_axis=2)
    return ssim