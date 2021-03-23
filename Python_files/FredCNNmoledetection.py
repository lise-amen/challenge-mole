# Name of the project: Mole Detection
# Period: from 17/03/2021 to 24/03/2021
# Context of the study: BeCode, Liège Campus, AI/Data Operator Bootcamp
# This file implements a convolutional neural network using the PyTorch framework. The aim is to classify mole images into three categories: (1) junctional, compound or dermal nevus, (2) atypical nevus and (3) melanoma
# Author: Frédéric Fourré
# Email: fourrefrederic593@gmail.com


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


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



# load data with PyTorch
# image : 3 x sizein x sizein
sizein = 64

# resize + totensor
transformc = transforms.Compose([transforms.Resize((sizein,sizein)), transforms.ToTensor()])

# put here your folder './folder' which contains subfolders './folder/classe1', './folder/classe2', './folder/classe3'
dat1 = datasets.ImageFolder('./folder', transform = transformc)

# check an image
img, label = dat1[200]
img.shape
torch.max(img), torch.min(img)
plt.figure(1)
plt.imshow(img.permute(1, 2, 0))
plt.show()


# class mynet

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        # two convolutional layers, kernel 3x3 + padding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        # reshape !
        out = out.view(-1, 16 * 16 * 16)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out
        
        

model = mynet()
# check model on an image
model(img.unsqueeze(0))


# training loop
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader: # loop on batches
            outputs = model(imgs) # feeds the model with a batch
            loss = loss_fn(outputs, labels) # computes the loss
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            loss_train += loss.item()
        
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))



# split dataset 
frac=0.7
lengths=[int(len(dat1)*0.7), int(len(dat1))-int(len(dat1)*0.7)]

train_load, val_load = torch.utils.data.random_split(dat1, lengths, generator = torch.Generator().manual_seed(42))

# set shuffle to True if SGD algorithm (see optimizer below)
# size of a batch
batchsi=64
train_loader1 = torch.utils.data.DataLoader(train_load, batch_size=batchsi, shuffle=True)

# number of epochs
nepoch=100

optimizer = optim.SGD(model.parameters(), lr=1e-2)

# loss function
loss_fn = nn.CrossEntropyLoss() 

# training 
training_loop(n_epochs = nepoch, optimizer = optimizer, model = model, loss_fn = loss_fn, train_loader = train_loader1)


# validation
train_loader2 = torch.utils.data.DataLoader(train_load, batch_size=batchsi, shuffle=False)
val_loader1 = torch.utils.data.DataLoader(val_load, batch_size=batchsi, shuffle=False)


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0 
        total = 0
        with torch.no_grad():
            for imgs, labels in loader: # loop on the batches
                outputs = model(imgs) 
                _, predicted = torch.max(outputs, dim=1) 
                total += labels.shape[0] 
                correct += int((predicted == labels).sum()) 
        print("Accuracy {}: {:.2f}".format(name , correct / total)) 

validate(model, train_loader2, val_loader1)


# validation with batch_size = 1
train_loader3 = torch.utils.data.DataLoader(train_load, batch_size=1, shuffle=False)
val_loader3 = torch.utils.data.DataLoader(val_load, batch_size=1, shuffle=False)

# train_loader3
model.eval()
with torch.no_grad():
    predclasst=[]
    trueclasst=[]
    for img, label in train_loader3:  
        output = model(img)        
        _, pred = torch.max(output, dim=1)                 
        predclasst.append(pred.item())
        trueclasst.append(label.item())

# observed and predicted values for each class (indicative)
valtruet, counttruet = np.unique(trueclasst, return_counts=True)
valpredt, countpredt = np.unique(predclasst, return_counts=True)

plt.figure(1)
plt.vlines(valtruet, 0, counttruet, colors='b', lw=10, label="TRUE")
plt.vlines(valpredt+0.1, 0, countpredt, colors='r', lw=10, label="PREDICTED")
plt.legend(loc="upper right")
plt.title('TRAIN')       

# val_loader3
with torch.no_grad():
    predclassv=[]
    trueclassv=[]
    for img, label in val_loader3:   
        output = model(img)       
        _, pred = torch.max(output, dim=1)                 
        predclassv.append(pred.item())
        trueclassv.append(label.item())

# observed and predicted values for each class (indicative)
valtruev, counttruev = np.unique(trueclassv, return_counts=True)
valpredv, countpredv = np.unique(predclassv, return_counts=True)

plt.figure(2)
plt.vlines(valtruev, 0, counttruev, colors='b', lw=10, label="TRUE")
plt.vlines(valpredv+0.1, 0, countpredv, colors='r', lw=10, label="PREDICTED")
plt.legend(loc="upper right")
plt.title('VAL')  
plt.show()

print(counttruet,countpredt)       
print(counttruev,countpredv)
print((countpredt-counttruet)/counttruet)
print((countpredv-counttruev)/counttruev)

# confusion matrix
print(confusion_matrix(trueclasst, predclasst))
print(confusion_matrix(trueclassv, predclassv))

print(sum(np.diag(confusion_matrix(trueclasst, predclasst)))/sum(sum(confusion_matrix(trueclasst, predclasst))))
print(sum(np.diag(confusion_matrix(trueclassv, predclassv)))/sum(sum(confusion_matrix(trueclassv, predclassv))))







