import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torchvision import datasets

import cv2


class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        # two convolutional layers, kernel 3x3 + padding
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 16 * 32, 64)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        # reshape !
        out = out.view(-1, 16 * 16 * 32)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out

def data_transform(img_path:str):

    sizein = 64
    
    image = cv2.imread(img_path)

    transformc = transforms.Compose([transforms.ToTensor(), transforms.Resize((sizein,sizein))])
    image = transformc(image)
    
    """
    t2 = transforms.ToTensor()
    image = t2(image)

    t1 = transforms.Resize((sizein,sizein))
    image = t1(image)
    """
    image = image.unsqueeze(0)

    return image
