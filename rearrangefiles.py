# BeCode AI Bootcamp
# Challenge 17/03/2021: Mole Detection
# This file allows to copy images from original directories 'SET_D', 'SET_E' and 'SET_F' to the new directories ./myfolder/i, where i = 1, 2 or 3 is the class (or category) of the image which will be used for classification. The user only needs to specify the name ./myfolder, the directories ./myfolder/i will be build automatically
# Contributor: Frédéric Fourré
 

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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



# directory after downloading images
datpath = './Mole_Data'
# bitmat images
print(os.listdir(datpath)) 
dir1 = os.listdir(datpath) 
dir2 = ['SET_D', 'SET_E', 'SET_F']
# list files in 'SET_E'
os.listdir(datpath + '/' + dir2[1])


# Pandas
dat = pd.read_csv('./Mole_Data/CLIN_DIA.csv')
dat.head()

dat.columns
dat.shape

list(dat.columns.values)
# ['klin. Diagn.', 'id', 'nr', 'Histo performed', 'Diagnose red.', 'kat.Diagnose']

dat.columns = ['klindiag', 'img', 'nr', 'histo', 'diagred', 'cat']
dat.head()


# shape (3000,2)
datimg = dat[['img','cat']]
datimg.loc[:,'img'] = datimg.img + '.bmp'
datimg.head()
datimg.isnull().sum()

datimg = datimg[(datimg.cat != '?')]


datimg.loc[:,'cat'] = datimg['cat'].astype(int)
print(datimg.dtypes)
datimg.cat.value_counts()
# 1 2300; 2 585; 0 62; 3 52; 2300+585+62+52 = 2999 rows

# a function to do this job would be better !
for i in range(len(dir2)):    
    images = os.listdir(datpath + '/' + dir2[i])
    srcp = datpath + '/' + dir2[i] + '/'
    print(i)
    for item in images:
        if item != 'Thumbs.db':
            x = item.lower()
            cat = datimg.loc[datimg.img == x].cat.item()
            # put here your destination folder
            dstp = './myfolder/' + str(cat) + '/'
            shutil.copy(srcp + item, dstp + item)




