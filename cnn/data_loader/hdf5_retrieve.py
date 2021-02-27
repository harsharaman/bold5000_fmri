#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import glob
import pickle
import joblib
import time
import numpy as np
import h5py


# In[41]:


source_dir = '/home/joshini/simlink_dataset/HDF5_trial/CSI1/'
file_list = glob.glob(source_dir + 'CSI1_*_X_*.h5')
file_list


# In[42]:


X , y = [],[]
start = time.time()
for num, file in enumerate(file_list):
    data = joblib.load(file)
    print(num)
    X.append(data)
#     y.append(data['y'])
    end = time.time()
print((end - start))


# In[44]:


X , y = [],[]
for hFile in file_list:
    start = time.time()
    with h5py.File(hFile, 'r') as f: 
        x_roi = f['X']
        X.append(x_roi)
        end = time.time()
    print((end - start))


# In[37]:


type(x_roi)


# In[34]:


os.getcwd()


# In[40]:


len(file_list)


# In[ ]:




