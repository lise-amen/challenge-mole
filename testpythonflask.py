#fichier pour entrer une seule image au format .bmp ou etc. dans le modèle PyTorch


import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import seaborn as sns
from IPython.display import display 

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from sklearn.metrics import confusion_matrix


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
        
        

loaded_model = mynet()

model_path='./'

filename='6987CNN2pixel64kernel3inputC643216epoch200.pt'

loaded_model.load_state_dict(torch.load(model_path + filename))

sizein=64

datpath='./upload'

filename='90d269.bmp'

image = cv2.imread(datpath + '/'+ filename) 

t2=transforms.ToTensor()

imgt2=t2(image)

print(imgt2.shape)

t1=transforms.Resize((sizein,sizein))

imgt1=t1(imgt2)

print(imgt1.shape)

img=imgt1.unsqueeze(0)

print(img.shape)

output=model(img)

fsoftmax = nn.Softmax()

proba=fsoftmax(output)

print(proba)

torch.max(proba)

_, indice = torch.max(proba, dim=1)  

print(indice.item())




