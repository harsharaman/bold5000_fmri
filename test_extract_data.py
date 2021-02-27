import os
import pandas as pd
from collections import defaultdict
import nibabel as nib
from glob import glob
import subprocess
import sys
from scipy.io import savemat
import numpy as np
import gc
import pickle
from numpy import savez_compressed
# Dicts holding training set and labels for each mask
X = defaultdict(list)
Y = defaultdict(list)
Ynames = defaultdict(list)

# Manual one-hot encoding
onehot = {'imagenet': [1,0,0,0],
          'rep_imagenet': [1,0,0,0],
          'coco': [0,1,0,0],
          'rep_coco': [0,1,0,0],
          'scenes': [0,0,1,0],
          'rep_scenes': [0,0,1,0],
          'none': [0,0,0,1]}

preproc_dir = "/home/ramanha/dataset/preprocessed/"          
#root = "/media/harsha/bigdaddy/hci/bold-preprocessed/trial_mask_file/sub-CSI2/"
mask_dir = "/home/ramanha/dataset/masks/"
subname = "sub-CSI3"
events_dir = "/home/ramanha/dataset/ds001499-download/"
data_dir = "/home/ramanha/dataset/cnn_data_test/"
#mask_files = glob(root + '/sub-*mask*.nii.gz')
#print(mask_files)

preproc_files = glob(preproc_dir + subname + '*_preproc.nii.gz')
#print(preproc_files)
# Resample mask, use first preproc file as representative sample
mask_resamp_file = "/home/ramanha/dataset/masks/sub-CSI3_mask-LHOPA-resamp.nii.gz"
#print(mask_resamp_file)
maskname = 'LHOPA'
mask = nib.load(mask_resamp_file).get_fdata(dtype=np.float32)
print(np.where(mask!=0))

for pnum, preproc in enumerate(preproc_files):
                X,Y,Ynames = [], [], []
                print('\t\tPreprocessed file %d out of %d' % ((pnum + 1), len(preproc_files)))
                #mask = nib.load(mask_resamp_file).get_fdata(dtype=np.float32)
                print('1. Checkpoint passed')
                items = preproc.split('_')
                print('2. Checkpoint passed')
              
                ses = items[-3]
                print('3. Checkpoint passed')
                
                run = items[-2]
                print('4. Checkpoint passed')
                
                event_file = glob(os.path.join(events_dir,subname,ses,'func','*' + run + '_events.tsv'))[0]
                print('5. Checkpoint passed')
                
                # Load events and image
                events = pd.read_csv(event_file, sep = '\t')
                print('6. Checkpoint passed')
                
                #img = nib.load(preproc).get_fdata(dtype=np.float32)
                img_proxy = nib.load(preproc)
                print('7. Checkpoint passed')
                #gc.collect()
                # Apply mask
                #img = np.transpose(img, axes=(3,0,1,2))
                print('8. Checkpoint passed')
                start_all, end_all = [], []
                for index, row in events.iterrows():
                    # Beginning TR of trial
                    start = int(round(row['onset']) / 2)
                    start_all.append(start)
                    #print('start: {}'.format(start))
                    # Ending TR of trial, start + 10 sec, or 5 TRs
                    end = start + 5
                    #print('end: {}'.format(end))
                    end_all.append(end)
                print(start_all[0], end_all[-1])
                img = img_proxy.dataobj[:,:,:,start_all[0]:end_all[-1]]
                print(type(img))
                img = np.transpose(img, axes=(3,0,1,2))
                #print(img.shape)
                #print(mask.shape)
                roi = np.where(mask==0, 0, img)
                print("ROI") # Shape: voxels x TRs
                print(roi.shape)
                assert img.shape == roi.shape
                print(sys.getrefcount(img))
                img_proxy.uncache()
                del img_proxy
                del img
                gc.collect()
                #del mask
                for index, row in events.iterrows():
                    # Beginning TR of trial
                    start = int(round(row['onset']) / 2)
                    # Ending TR of trial, start + 10 sec, or 5 TRs
                    end = start + 5
                    #print(start,end)
                    x = roi[start:end,:,:,:]#.T
                    y = onehot[row['ImgType']]
                    X.append(x) # Big X should be of shape (samples, timepoints, features)
                    Y.append(y)
                    Ynames.append(row['ImgName'])
                # Save last ten TRs as no stimulus, if enough data is left
                if roi.shape[0] - end >= 5:
                    x = roi[end:end+5,:,:,:]#.T
                    y = onehot['none']
                    X.append(x)
                    Y.append(y)
                    Ynames.append('none')

                with open(data_dir + 'X_test_CSI3_LHOPA_' + str(pnum) + '.p', 'wb') as f:
                        pickle.dump(X, f)

                with open(data_dir + 'Y_test_CSI3_LHOPA_' + str(pnum) + '.p', 'wb') as f:
                        pickle.dump(Y, f)

                with open(data_dir + 'Ynames_test_CSI3_LHOPA_' + str(pnum) + '.p', 'wb') as f:
                        pickle.dump(Ynames, f)
                del X
                del Y
                del Ynames
                del roi
                gc.collect()
                




'''
#with open(data_dir + 'X_test_LHOPA.p', 'wb') as f:
   # pickle.dump(X, f)
    
with open(data_dir + 'Y_test_LHOPA.p', 'wb') as f:
    pickle.dump(Y, f)
    
with open(data_dir + 'Ylabels_test_LHOPA.p', 'wb') as f:
    pickle.dump(Ynames, f)
'''
