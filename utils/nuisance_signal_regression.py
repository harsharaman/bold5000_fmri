import numpy as np
import os
import sys
from glob import glob
import pandas as pd
import subprocess
from scipy.io import savemat
import argparse

'''
First, we must regress out nuisance signals. In fMRI analysis, many nuisance signals are gathering from the processing pipeline and are linearly regressed or detrended from the whole brain timeseries. Some commonly used nuisance signals include motion parameters (rigid body motion involves three translation, three rotation parameters), average CSF signal, average white matter signal, and all their derivatives. The regression of the global signal - or the mean signal across all brain voxels - is highly contended in the fMRI field. Many consider it to be physiological noise, others consider it to contain important neural information. For our purposes, we will not regress out global signal. Our goal is to decode visual stimuli features from fMRI timeseries of visual areas of the brain.
NOTE: make sure to install FSL by running fsinstaller.py and change the folder path accordingly in line 143.

PARAMETERS

regressors_dir: an output directory to hold the nuisance regressors text files
preproc_dir: an output directory for the fully processed subject data
'''

parser = argparse.ArgumentParser()

parser.add_argument('--regressors_dir', type=str, metavar='N',
				help="an output directory to hold the nuisance regressors text files")
parser.add_argument('--preproc_dir', type=str, metavar='N',
				help = "an output directory for the fully processed subject data")

args = parser.parse_args()

def get_censored_frames(subj_df, threshold):
        '''
        Takes a subject dataframe loaded from the regressors.tsv file and extracts FD.
        Performs motion scrubbing by marking all volumes with FD greater than threshold with 1.
        Additionally marks one volume before and two after. Converts to a sparse matrix where
        each column has one frame that is censored.
        '''
	
        # Threshold should be bounded between 0 and 1
        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold should be bounded between 0 and 1.')
            
        # Extract FD column
        fd = subj_df['FramewiseDisplacement'].values
        fd = np.nan_to_num(fd)
        
        # Create censor vector
        censor = [0 if m <= threshold else 1 for m in fd]
        
        # Censor one back, two forward
        censor_fixed = np.zeros_like(censor)
        for ind,c in enumerate(censor):
            if c == 1:
                try:
                    censor_fixed[ind-1:ind+3] = 1
                except IndexError:
                    censor_fixed[ind-1:] = 1
                    
        #Convert to sparse matrix
        censor_feat = np.zeros((censor_fixed.shape[0], np.count_nonzero(censor_fixed)))
        col = 0
        for ind,c in enumerate(censor_fixed):
            if c == 1:
                censor_feat[ind,col] = 1
                col +=1

        return censor_feat, censor_fixed
                                                                                                                                                                                   
def get_regressors(subj_df, regressors):
        """
        Takes a subject dataframe loaded from the regressors.tsv file and extracts relevant regressors (list)
        """
	
        # Should be of dim TRs x # regressors
        regress_mat = np.array([subj_df[regressor].values for regressor in regressors]).T
        
        # Calculate derivatives manually
        deriv = np.diff(regress_mat,axis=0)
        deriv = np.insert(deriv, 0, regress_mat[0], axis = 0)
        final = np.hstack((regress_mat,deriv))
        
        return final

def get_subj_dirs(fmriprep_dir):
        """
        Returns subject directories from fmriprep directory
        """
	
        subj_dirs = [f for f in os.listdir(fmri_dir) if os.path.isdir(os.path.join(fmri_dir, f)) and 'sub' in f]
        return subj_dirs

fd_threshold = 0.5 #Threshold of FD for censoring a frame

#All the nuisance signals we wish to remove. Derivatives are not included, these are calculated manually

regressors = ['CSF', 'WhiteMatter', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

#Set directories
fmri_dir, nuisance_dir, regressors_dir, preproc_dir, standard = ['/bigpool/export/users/datasets_faprak2020/BOLD5000/ds001499-download', '/bigpool/export/users/datasets_faprak2020/BOLD5000/ds001499-download/derivatives/fmriprep', args.regressors_dir, args.preproc_dir, '/bigpool/export/users/datasets_faprak2020/BOLD5000/MNI152_T1_2mm_brain.nii.gz']

subj_dirs = get_subj_dirs(fmri_dir)

#Loop through each subject and get regressors, perform scrubbing
for subj in sorted(subj_dirs):
    print('Processing %s' % subj)
    #Absolute path of current subject
    subj_dir_abs = os.path.join(fmri_dir, subj)
    sess_dirs = sorted([f for f in os.listdir(subj_dir_abs) if os.path.isdir(os.path.join(subj_dir_abs, f)) and 'ses-' in f])
    if not sess_dirs:
        #If there are not multiple sessions, then set to list of empty string to tierate only once in for loop
        sess_dirs = ['']

    for sessnum, sess in enumerate(sess_dirs):
        print('\tSession %d out of %d' % ((sessnum + 1), len(sess_dirs)))
        #Absolute path of current session
        sess_dir_abs = os.path.join(subj_dir_abs, sess)
        conf_sess_dir_abs = os.path.join(nuisance_dir, subj, 'ses-' + str(sessnum+1).zfill(2))
        bold_files = sorted(glob(sess_dir_abs + '/func/*task-5000scenes*bold.nii.gz'))
        confound_files = sorted(glob(conf_sess_dir_abs + '/func/*task-5000scenes*confounds*.tsv'))
        
        #For multiple runs
        for runnum, (bold, confound) in enumerate(zip(bold_files, confound_files)):
            print("\t\tRun %d out of %d" % ((runnum + 1), len(bold_files)))
            df = pd.read_csv(confound, sep='\t')
            censor_mat, censor_frames = get_censored_frames(df, fd_threshold)
            regress_mat = get_regressors(df, regressors)
            nuisance_mat = np.hstack((censor_mat, regress_mat))
            prefix = os.path.join(regressors_dir, subj + '_ses-' + str(sessnum+1).zfill(2) + '_run-' + str(runnum+1).zfill(2) + '_')
            outfile = prefix + 'censored.txt'
            outfile = prefix + 'nuisance_regressors.txt'

            #Perform registration
            bold_reg = '/bigpool/export/users/datasets_faprak2020/BOLD5000/registered' + bold[55:-7] + '_registered.nii.gz'
            
            outmat = '/bigpool/export/users/datasets_faprak2020/BOLD5000/registered/' + subj + '_ses-' + str(sessnum+1).zfill(2) + '_run-' + str(runnum+1).zfill(2) + '_native2template.mat'

            cmd = 'flirt -in ' + bold + ' -ref ' + standard + ' -omat ' + outmat
            subprocess.call(cmd, shell=True)
            cmd = 'flirt -in ' + bold + ' -ref ' + standard + ' -out ' + bold_reg + ' -init ' + outmat + ' -applyxfm -v'
            subprocess.call(cmd, shell=True)

            #Use AFNI to perform regression
            outfile = outfile[:-3] + 'mat'
            savemat(outfile, {'nuisance_regressors': regress_mat})
            subprocess.call('python2 /home/harsha/abin/read_matlab_files.py -infiles ' + outfile + ' -prefix ' + prefix[:-1] + ' -overwrite', shell = True)
            design = glob(prefix[:-1] + '*.1D')[0]
            prefix = os.path.join(preproc_dir, subj + '_ses-' + str(sessnum+1).zfill(2) + '_run-' + str(runnum+1).zfill(2) + '_')
            outfile = prefix + 'preproc.nii.gz'
            subprocess.call('3dTproject -input ' + bold_reg + ' -prefix ' + outfile + ' -ort ' + design + ' -polort 2 -passband 0.009 0.1 -blur 6 -quiet -overwrite', shell = True)
            
            
            

