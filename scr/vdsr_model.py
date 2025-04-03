""" This file contains the VDSR model for imagesuper-resolution.
The functions for training and inference is also included here. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import Shuffler
from tqdm import tqdm
import torchvision.models as models
from torchvision.models import VGG19_Weights

class ConvReLU(nn.Module):
    def __init__(self, channels):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self, channels=64, input_channels=1, num_layers=18, ds_factor=1, residual_rate=1):
        super(VDSR, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.ds_factor = ds_factor
        self.residual_rate = residual_rate

        self.conv_in = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_layers = self.make_layers()

        self.conv_out = nn.Conv2d(channels, input_channels, kernel_size=3, padding=1, bias=False)

    def make_layers(self):
        layers = nn.Sequential(*[ConvReLU(self.channels) for _ in range(self.num_layers)])
        return layers
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.ds_factor, mode='bicubic', align_corners=True)
        residual = x
        out = self.relu(self.conv_in(x))
        out = self.conv_layers(out)
        out = self.conv_out(out)
        out = torch.add(out,residual*self.residual_rate)  # Residual Learning: Add input back
        return out

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, sr, hr):
        return torch.mean(torch.sqrt((sr-hr)**2 + self.epsilon**2))

class VGGLoss(nn.Module):
    def __init__(self, layer='relu5_4'):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.layers = {
            'relu1_2': 3,  'relu2_2': 8,  'relu3_4': 17,
            'relu4_4': 26, 'relu5_4': 35
        }
        assert layer in self.layers, f"Invalid layer {layer}, choose from {list(self.layers.keys())}"

        self.vgg = nn.Sequential(*list(vgg[:self.layers[layer]])).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG model

        self.criterion = nn.L1Loss()  # Use L1 loss for perceptual similarity

    def forward(self, sr, hr):
        sr_vgg = self.vgg(sr)
        hr_vgg = self.vgg(hr)
        return self.criterion(sr_vgg, hr_vgg)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_VDSR(model, epochs, dataset, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    mse_loss = nn.MSELoss()
    L1_loss = CharbonnierLoss()
    # vgg_loss = VGGLoss(layer='relu5_4').to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(epochs):
        suffled_dataset = Shuffler(dataset, buffer_size=10)
        dataloader = DataLoader(suffled_dataset, batch_size)
        model.train()
        epoch_loss = 0.0
        index = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        for lr, hr in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)

            loss_mse = mse_loss(sr, hr)
            loss_L1 = L1_loss(sr, hr)
            # loss_vgg = vgg_loss(sr, hr)
            
            loss = 0.5*loss_mse + 0.5*loss_L1 # + 0.2*loss_vgg

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_loss += loss.item()
            index += 1
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
        scheduler.step()
        
        avg_loss = epoch_loss/index
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return model

class PnP_ADMM:
    def __init__(self, model, rho=0.1, num_iters=10, ds_factor=2):
        self.model = model  # VDSR as the denoiser
        self.rho = rho
        self.num_iters = num_iters
        self.ds_factor = ds_factor

    def forward(self, lr):
        # Initialize Variables
        up_lr = self.model(lr)
        z = up_lr.clone()       
        u = torch.zeros_like(lr)  # Dual variable initialized to zero

        for _ in range(self.num_iters):
            # Solve Least Squares Problem (Data Consistency)
            z_down = F.interpolate(z, scale_factor=1/self.ds_factor, mode='bicubic', align_corners=False)
            residual = lr - z_down  # Difference between LR input and downsampled SR estimate
            sr = z_down + self.rho * residual
            # Apply Denoiser (PnP Step)
            z = self.model(sr + u)
            # Align the scale for update
            s = F.interpolate(z, scale_factor=1/self.ds_factor, mode='bicubic', align_corners=False)
            # Update Dual Variable
            u = u + (sr - s)
        return z  
    
def inference_VDSR(model, dataset, pnp:bool=False, num_iters=10):
    model.eval()
    data = DataLoader(dataset, 1, shuffle=False)
    lr_list = []
    sr_list = []
    hr_list = []
    index = 0
    for lr, hr in data:
        with torch.no_grad():
            if pnp:
                do_pnp = PnP_ADMM(model, rho=0.1, num_iters=num_iters, ds_factor=model.ds_factor)
                sr = do_pnp.forward(lr)
            else:    
                sr = model(lr)

        sr = sr.squeeze(0).permute(1, 2, 0).numpy()
        sr = sr.clip(0, 255).astype("uint8")

        lr = lr.squeeze(0).permute(1, 2, 0).numpy()
        lr = lr.clip(0, 255).astype("uint8")

        hr = hr.squeeze(0).permute(1, 2, 0).numpy()
        hr = hr.clip(0, 255).astype("uint8")

        lr_list.append(lr)
        sr_list.append(sr)
        hr_list.append(hr)
        index+=1
        print(f'The image {index} has been successfully resolved!')
    return lr_list, sr_list, hr_list