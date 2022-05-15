import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
  
  
    def forward(self, input):
        out = F.relu(F.max_pool2d(self.conv1(input), 2))
        out = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(out)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        out = out.view(-1, 320)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return torch.sigmoid(out)
    


