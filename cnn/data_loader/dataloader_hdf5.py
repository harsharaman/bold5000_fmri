import torch
from torch.utils import data
import os
import numpy as np
import h5py

class fMRICNNcustomDataset(data.Dataset):
    """
    Args:
    fmri_data (string): Path to the fmri file with masks.
    fmri_label (string): Path to respective label of fmri (to which class it belongs to)
    fmri_labelname (string): Path to respective labelname of fmri
    transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_folder):
        self.path_list_X, self.path_list_Y = [] , []
        self.root_folder = data_folder
        #self.sub_list = ['CSI!', 'CSI2', 'CSI3', 'CSI4']
        for path, subdirs, files in os.walk(self.root_folder):
            for name in files:
                if '_X_' in name:
                    self.path_list_X.append(os.path.join(path,name))
                elif '_Y_Cluster_' in name:
                    self.path_list_Y.append(os.path.join(path,name))


    def __len__(self):
        'denotes the total number of samples'
        return len(self.path_list_X)

    def __getitem__(self, index):
        'Generates one sample of data'
        if torch.is_tensor(index):
            idx = idx.tolist()

        with h5py.File(self.path_list_X[index], 'r') as f:
            x_data = np.array(f['X'])

        with h5py.File(self.path_list_Y[index], 'r') as f:
            y_cluster = np.array(f['y_cluster'])
        return x_data, y_cluster

#obj = fMRICNNcustomDataset("/home/ramanha/dataset/traindata_cnn/")
#print(len(obj))
