# File to check the PyTorch CNN model on a single mole image 
# To execute this file: exec(open('testpythonflask.py').read()) 
# Contributor: Frédéric Fourré

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms


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

datpath='./pytorchdat/3'

# images in classe 3: D269, D288, D322, D328, D329, D340, D353, D437, D540
imgname='D540.BMP'

# read image with cv2
#image = cv2.imread(datpath + '/' + imgname, cv2.IMREAD_COLOR)
#imagec = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# read the image with PIL
imagec = Image.open(datpath + '/' + imgname)

plt.imshow(imagec)
plt.show()

t2=transforms.ToTensor()

imgt2=t2(imagec)

print(imgt2.shape)

sizein=64
t1=transforms.Resize((sizein,sizein))

imgt1=t1(imgt2)

print(imgt1.shape)

img=imgt1.unsqueeze(0)

print(img.shape)

output=loaded_model(img)

fsoftmax = nn.Softmax()

proba=fsoftmax(output)

print(proba)

torch.max(proba)

_, indice = torch.max(proba, dim=1)  

print(indice.item())




