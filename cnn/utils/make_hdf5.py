#!/usr/bin/env python
# coding: utf-8

# In[28]:


import h5py
import os
import torch


# In[16]:


h5_file = h5py.File("/home/ramanha/dataset/traindata_cnn_dict/CSI3testfile.hdf5", "a")
#print(h5_file['CSI3']['Y'].shape)


# In[11]:


data_folder = "/home/ramanha/dataset/traindata_cnn_dict/CSI3/"
csi3 = h5_file.create_group("CSI3")
datasetX = csi3.create_dataset("X", (1,5,91,109,91), maxshape=(None, 5,91,109,91))
datasetY = csi3.create_dataset("Y", (1,), maxshape=(None,))


# In[ ]:


file_list, X, y = [], [],[]
for root, dire, files in os.walk(data_folder):
    for idx, fil in enumerate(files):
        f = torch.load(os.path.join(root,fil))
        print(fil)

        #with h5py.File("/home/ramanha/dataset/traindata_cnn_dict/CSI3testfile.hdf5", "a") as h5_file:
            #print(f['X'].shape)
        datasetX[:,:,:,:] = f['X']
        datasetY[:] = f['y']
        datasetX.resize(idx+1, axis=0)
        datasetY.resize(idx+1, axis=0)


            
    
#print(len(X), len(y))


# In[25]:


#csi3 = h5_file.create_group("CSI3")


# In[26]:


#csi2 = h5_file.create_group("CSI2")


# In[ ]:


#datasetX = csi3.create_dataset("X", (1,5,91,104,91), maxshape=(None, 5,91,104,91)
#datasetY = csi3.create_dataset("Y", data=(1,1), maxshape=(None,1))


# In[ ]:


#file_list = glob.glob(data_folder + 'CSI1_*.p')
#print(file_list)


# In[ ]:




