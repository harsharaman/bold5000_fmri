import torch
from torch.utils import data
import os
import numpy as np

class fMRICNNcustomDataset(data.Dataset):
    """
    Args:
    fmri_data (string): Path to the fmri file with masks.
    fmri_label (string): Path to respective label of fmri (to which class it belongs to)
    fmri_labelname (string): Path to respective labelname of fmri
    transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_folder):
        self.path_list = []
        self.root_folder = data_folder
        self.sub_list = ['CSI!', 'CSI2', 'CSI3', 'CSI4']
        for path, subdirs, files in os.walk(self.root_folder):
            for name in files:
                if '.p' in name:
                    self.path_list.append(os.path.join(path,name))

    def __len__(self):
        'denotes the total number of samples'
        return len(self.path_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        if torch.is_tensor(index):
            idx = idx.tolist()
        
        x_data = np.array(torch.load(self.path_list[index])['X'])
        y_cluster = np.array(torch.load(self.path_list[index])['y'])
        return x_data, y_cluster
