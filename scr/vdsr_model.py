""" This file contains the VDSR model for imagesuper-resolution.
The functions for training and inference is also included here. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import Shuffler
from tqdm import tqdm

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
            
            loss = 0.5*loss_mse + 0.5*loss_L1

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

def inference_VDSR(model, dataset):
    model.eval()
    data = DataLoader(dataset, 1, shuffle=False)
    lr_list = []
    sr_list = []
    hr_list = []
    index = 0
    for lr, hr in data:
        lr = lr
        hr = hr
        index += 1
        with torch.no_grad():
            sr_image = model(lr)

        sr = sr_image.squeeze(0).permute(1, 2, 0).numpy()
        sr = sr.clip(0, 255).astype("uint8")

        lr = lr.squeeze(0).permute(1, 2, 0).numpy()
        lr = lr.clip(0, 255).astype("uint8")

        hr = hr.squeeze(0).permute(1, 2, 0).numpy()
        hr = hr.clip(0, 255).astype("uint8")

        lr_list.append(lr)
        sr_list.append(sr)
        hr_list.append(hr)

        print(f'The image number {index} is resolved.')

    return lr_list, sr_list, hr_list