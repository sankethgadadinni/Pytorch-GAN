import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.latent_dim = config.latent_dim
        
        self.linear = nn.Linear(self.latent_dim, 7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)
        
    
    def forward(self, input):
        
        out = self.linear(input)
        out = F.relu(out)
        
        out = out.view(-1, 64, 7, 7)
        
        out = self.ct1(out)
        out = F.relu(out)
        
        out = self.ct2(out)
        out = F.relu(out)
        
        out = self.conv(out)
        
        return out
    
        