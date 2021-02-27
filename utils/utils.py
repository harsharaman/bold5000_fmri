import numpy as np
import os
import sys
from glob import glob
import pandas as pd
import subprocess
from scipy.io import savemat

'''
Take a subject dataframe loaded from regressors.tsv file and extract FD.
Performs motion scrubbing by marking all volumes with FD grater than threshold with 1.
Additionally marks one volume before and two after. Converts to a sparse matrox where each 
column has one frame that is censored
'''

def get_censored_frames(sub_df, theshold):
    #Threshold has limits (0,1)
    if threshold < 0 or threshold > 1:
        raise ValueError('Threshold should be bounded between 0 and 1.')

    #Extract FD column
    fd = sub_df['FramewiseDisplacement'].values
    fd = np.nan_to_num(fd)

    #Create censor vector
    censor = [0 if m <= threshold else 1 for m in fd]

    #Censor one back, two forward
    censor_fixed = np.zeros_like(censor)

    for ind, c in enumerate(censor):
        if c == 1:
            try:
                censor_fixed[ind-1:ind+3] = 1
            except IndexError:
                censor_fixed[ind-1:] = 1

    #Convert to sparse matrix
    censor_feat = np.zeros((censor_fixed.shape[0], np.count_nonzero(censor_fixed)))

    col=0

    for ind, c in enumerate(censor_fixed):
        if c == 1:
            censor_feat[ind, col] = 1
            col += 1

    return censor_feat, censor_fixed

def get_regressors(sub_df, regressors):
    '''
    Takes a subject dataframe loaded from the regressors.tsc file and extracts relevant regressors (list)
    '''
    #Should be of dimensions TRs x #regressors
    regress_mat = np.array([sub_df[regressor].values for regressor in regressors]).T

    #Calculate derivatives manually
    deriv = np.diff(regress_mat, axis=0)
    deriv = np.insert(deriv, 0, regress_mat[0], axis=0)
    final = np.hstack((regress_mat, deriv))

    return final

def get_subj_dirs(fmriprep_dir):
    '''
    Returns subject directories from fmriprep directory
    '''
    sub_dirs = [f for f in os.listdir(fmri_dir) if os.path.isdir(os.path.join(fmri_dir, f)) and 'sub' in f]
    return sub_dirs
