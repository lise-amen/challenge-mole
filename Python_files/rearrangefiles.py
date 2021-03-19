# BeCode AI Bootcamp
# Challenge 17/03/2021: Mole Detection
# This file allows to copy images from original directories 'SET_D', 'SET_E' and 'SET_F' to the new directories ./myfolder/i, where i = 1, 2 or 3 is the class (or category) of the image which will be used for classification. 
# The user only needs to specify the name ./myfolder, the directories ./myfolder/i will be built automatically
# Contributor: Frédéric Fourré
 

import os
import shutil
import pandas as pd


# directory after downloading images
datpath = '../data/Mole_Data/'
# bitmat images
print(os.listdir(datpath)) 
dir1 = os.listdir(datpath) 
dir2 = ['SET_D', 'SET_E', 'SET_F']
# list files in 'SET_E'
os.listdir(datpath + '/' + dir2[1])

# Pandas
dat = pd.read_csv('../data/Mole_Data/CLIN_DIA.csv')

dat.columns = ['klindiag', 'img', 'nr', 'histo', 'diagred', 'cat']

# shape (3000,2)
datimg = dat[['img','cat']]
datimg.loc[:,'img'] = datimg.img + '.bmp'

datimg = datimg[(datimg.cat != '?')]

datimg.loc[:,'cat'] = datimg['cat'].astype(int)
"""
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
            dstp = '../data/Mole_Data_Rearranged' + str(cat) + '/'
            shutil.copy(srcp + item, dstp + item)
"""