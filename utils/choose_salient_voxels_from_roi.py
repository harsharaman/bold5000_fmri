from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse

'''
Since each subject's ROI mask has a different number of voxels and thus a different number of features, it 
would be difficult to concatenate them. Let's take a look at which voxels re highly correlated with the labels,
and maybe we can choose the voxels based on that
'''

parser.add_argument('--data_dir', type=str, metavar='N', 
                                help = "the folder where the training data to be saved")
args = parser.parse_args()
data_dir = args.data_dir

with open(data_dir + 'X_unfixed.p', 'r') as f:
    X = pickle.load(f)

with open(data_dir + 'Y_unfixed.p', 'r') as f:
    Y = pickle.load(f)
    
with open(data_dir + 'Ylabels_unfixed.p', 'r') as f:
    Ynames = pickle.load(f)

# A priori known number of voxels extracted from each subject for the LHPPA
num_voxels = [172, 131, 112, 157]
for n in num_voxels:
    sub_timeseries, sub_labels = zip(*[(arrx,np.repeat(np.array(arry)[np.newaxis,:], 5, axis=0)) for arrx,arry in zip(X['LHPPA'], Y['LHPPA']) if arrx.shape[1] == n])
    sub_timeseries = np.concatenate(sub_timeseries, axis=0)
    sub_labels = np.concatenate(sub_labels, axis = 0)
    sub_labels = np.array([np.nonzero(label)[0][0] + 1 for label in sub_labels])
    sub_labels[sub_labels == 4] = 0
    corr,p = zip(*[pearsonr(voxel,sub_labels) for voxel in sub_timeseries.T])
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(corr, color = 'r', marker = 'd')
    ax.set_xlabel('Voxel')
    ax.set_ylabel('Correlation with class labels')
    plt.show()
    sig = np.array(p) < 0.05
    print('Number of significant voxels: %d out of %d' % (np.count_nonzero(sig), sub_timeseries.shape[1]))

for mask in X.keys():
    last = 0
    print(mask)
    for sample in X[mask]:
        if sample.shape[1] != last:
            print('\t' + str(sample.shape))
        last = sample.shape[1]
        
#We will take the top x most correlated voxels from each subject from each mask, where x is some number less
#than all the subjects' number of voxels for that specific mask, but not so low that we filter out all the
#voxels.

topvoxels = {'LHPPA': 100,
             'RHLOC': 170,
             'LHLOC': 130,
             'RHEarlyVis': 220,
             'RHRSC': 100,
             'LHOPA': 70,
             'RHPPA': 140,
             'LHEarlyVis': 190,
             'LHRSC': 30,
             'RHOPA': 80}
             
X_new = {}

for mask in X:
    # Get subject specific number of voxels to identify them
    last = 0
    num_voxels = []
    x = []
    for sample in X[mask]:
        if sample.shape[1] != last:
            num_voxels.append(sample.shape[1])
        last = sample.shape[1]
    for n in num_voxels:
        sub_timeseries, sub_labels = zip(*[(arrx,np.repeat(np.array(arry)[np.newaxis,:], 5, axis=0)) for arrx,arry in zip(X[mask], Y[mask]) if arrx.shape[1] == n])
        sub_timeseries2 = np.concatenate(sub_timeseries, axis=0)
        sub_labels = np.concatenate(sub_labels, axis = 0)
        sub_labels = np.array([np.nonzero(label)[0][0] + 1 for label in sub_labels])
        sub_labels[sub_labels == 4] = 0
        corr,p = zip(*[pearsonr(voxel,sub_labels) for voxel in sub_timeseries2.T])
        # Get indices of top x correlated voxels for this mask
        idx = np.argsort(np.abs(corr))[::-1][:topvoxels[mask]]
        x.append(np.array(sub_timeseries)[:,:,idx])
    x = np.concatenate(x,axis=0)
    X_new[mask] = x
    
for mask in Y:
    Y[mask] = np.array(Y[mask])
    Ynames[mask] = np.array(Ynames[mask])
    
    print(mask + ': shape of X is ' + str(X_new[mask].shape))
    print(mask + ': shape of Y is ' + str(Ynames[mask].shape))
    
    np.save(data_dir + 'X_' + mask + '.npy', X_new[mask])
    np.save(data_dir + 'Y_' + mask + '.npy', Y[mask])
    np.save(data_dir + 'Ylabels_' + mask + '.npy', Ynames[mask])
    
with open(data_dir + 'X_fixed.p', 'wb') as f:
    pickle.dump(X_new, f)
    
with open(data_dir + 'Y_fixed.p', 'wb') as f:
    pickle.dump(Y, f)
    
with open(data_dir + 'Ylabels_fixed.p', 'wb') as f:
    pickle.dump(Ynames, f)




