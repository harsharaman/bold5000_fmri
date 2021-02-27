import os
import pandas as pd
from collections import defaultdict
import nibabel as nib
from glob import glob
import subprocess
import sys
from scipy.io import savemat
import numpy as np
import argparse

"""
We must extract the data of brain regions that are involved in visual processing. We will use the visual regions defined in the HCP MMP 1.0 atlas. We will take these binary masks and mutiply them to each subject's fMRI data to extract the information we need.

Also, we need to label each part of the data with the image it corresponds to and package the resulting data into neat X and Y matrices.

PARAMETERS - BE SURE TO SET THESE!!

roi_dir: where the roi_masks are located, should contain subject folders
preproc_dir: where the fully processed data is stored
events_dir: where the event files are held, should contain subject folders.
            This is probably the same folder as the original dataset/fmriprep folder.
data_dir: the output of where you want the training data to be saved
mask_dir: the output of where you want your resampled mask to be
            
"""
parser = argparse.ArgumentParser()

parser.add_argument('--preproc_dir', type=str, metavar='N',
				help = "an input directory for the fully processed subject data")
parser.add_argument('--events_dir', type=str, metavar='N', default="/bigpool/export/users/datasets_faprak2020/BOLD5000",
                                help = "where the event files are held, should contain subject folders. This is probably the same folder as the original dataset/fmriprep folder")
parser.add_argument('--data_dir', type=str, metavar='N', 
                                help = "the output of where you want the training data to be saved")
parser.add_argument('--mask_dir', type=str, metavar='N',
                                help = "the output of where you want your resampled mask to be")
parser.add_argument('--roi_dir', type=str, metavar='N', default="/bigpool/export/users/datasets_faprak2020/BOLD5000/ds001499-download/derivatives/spm", help = "where the roi_masks are located, should contain subject folders")

args = parser.parse_args()
                                                 
preproc_dir, events_dir, roi_dir, data_dir, mask_dir = [args.preproc_dir,
                                            args.events_dir,
                                            args.roi_dir,
                                            args.data_dir,
                                            args.mask_dir
                                            ]
                                            
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

# Walk through ROI mask directory
for root, dirs, files in os.walk(roi_dir):
    # If in a subject folder
    if 'sub' in root:
        subname = root.split('/')[-1]
        print(subname)
        # Gather all mask NIFTIS
        mask_files = glob(root + '/sub-*mask*.nii.gz')
        for mask_file in mask_files:
            maskname = mask_file.split('-')[-1].split('.')[0]
            print('\t' + maskname)
            # There are many runs and sessions per subject
            preproc_files = glob(preproc_dir + subname + '*_preproc.nii.gz')
            # Resample mask, use first preproc file as representative sample
            mask_resamp_file = mask_dir + mask_file.split('/')[-1][:-7] + '-resamp.nii.gz'
            subprocess.call('3dresample -master ' + preproc_files[0] + ' -prefix ' + mask_resamp_file + ' -input ' + mask_file, shell = True)
            # Load new mask file
            mask = nib.load(mask_resamp_file).get_fdata()
            for pnum, preproc in enumerate(preproc_files):
                print('\t\tPreprocessed file %d out of %d' % ((pnum + 1), len(preproc_files)))
                items = preproc.split('_')
                ses = items[-3]
                run = items[-2]
                event_file = glob(os.path.join(events_dir,subname,ses,'func','*' + run + '_events.tsv'))[0]
                # Load events and image
                events = pd.read_csv(event_file, sep = '\t')
                img = nib.load(preproc).get_fdata()
                # Apply mask
                img = np.reshape(img, (img.shape[0]*img.shape[1]*img.shape[2], -1))
                mask_fixed = mask.astype(bool).flatten()
                roi = img[mask_fixed] # Shape: voxels x TRs
                # Get relevant time intervals and labels from events file
                for index, row in events.iterrows():
                    # Beginning TR of trial
                    start = int(round(row['onset']) / 2)
                    # Ending TR of trial, start + 10 sec, or 5 TRs
                    end = start + 5
                    x = roi[:,start:end].T
                    y = onehot[row['ImgType']]
                    X[maskname].append(x) # Big X should be of shape (samples, timepoints, features)
                    Y[maskname].append(y)
                    Ynames[maskname].append(row['ImgName'])
                # Save last ten TRs as no stimulus, if enough data is left
                if roi.shape[1] - end >= 5:
                    x = roi[:,end:end+5].T
                    y = onehot['none']
                    X[maskname].append(x)
                    Y[maskname].append(y)
                    Ynames[maskname].append('none')

import pickle

with open(data_dir + 'X_unfixed.p', 'w') as f:
    pickle.dump(X, f)
    
with open(data_dir + 'Y_unfixed.p', 'w') as f:
    pickle.dump(Y, f)
    
with open(data_dir + 'Ylabels_unfixed.p', 'w') as f:
    pickle.dump(Ynames, f)
