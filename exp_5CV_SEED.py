# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:57:24 2020

@author: dykua

Main script for training with SEED data
For the benchmark purpose - 5cv
"""
#%%
import argparse
from re import sub
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from Utils import scores
from Models import *
# from visual import plot_confusion_matrix
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"   # use this line to select gpu '0' or '1'

#%%
# path = r'E:\Datasets\SEED'
# path = '/media/dykuang/SATA/Datasets/SEED'
path = '/mnt/HDD/Datasets/SEED'

parser = argparse.ArgumentParser()
parser.add_argument('--subject', help='subject index')
args = parser.parse_args()

subject = '{:02d}'.format( int(args.subject) )

X = loadmat( os.path.join(path, 'S{}_E01.mat'.format(subject)) )['segs'].transpose([2,1,0])
Y = loadmat( os.path.join(path, 'Label.mat') )['seg_labels'][0]

'''
exp paramters
'''
chns=62
nn_choice = 1

'''
Training paramers
'''
epochs = 80
batch_size = 128


nn_token = 'DC' # change this accordingly with the selection of network

# model = D2ANet(nb_classes = 3, Dist_M = Dist_M, penalty_rate=10, num_kpts=11,
#                Chans = chns, Samples = 200, 
#                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
#                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
#                learning_rate = 1e-2)

# model = EEGNet(nb_classes = 3, Chans = chns, Samples = 200, 
#                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
#                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
#                learning_rate = 1e-2)

# model = KANet(nb_classes = 3, Chans = chns, Samples = 200, 
#                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
#                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
#                learning_rate = 1e-2)

# model = MTNet(nb_classes = 3, Chans = chns, Samples = 200, 
#                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
#                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
#                penalty_rate=1.0, mono_mode='ID',
#                learning_rate = 1e-2)

# model = CANet(nb_classes = 3, Chans = chns, Samples = 200, attention_module = 'cbam_block',
#                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
#                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
#                learning_rate = 1e-2)

model = DeepConvNet(nb_classes = 3, Chans = chns, Samples = 200,
                    dropoutRate = 0.25,
                    optimizer = Adam, learning_rate = 1e-2)
    
print('#'*50)
print('CV on model {} with {} channels for subject {}.'.format(nn_token, chns, subject)) 
print('#'*50)
       
model.summary()
model.save_weights('model_ini.h5') # save an intial copy to reload at each fold

#%%
from sklearn.model_selection import StratifiedKFold, train_test_split

# using the same validation set
indices = np.arange(len(Y))
_, _, Ycv, Yval, CV_ind, val_ind = train_test_split(X[...,0], Y, indices, test_size=0.1667, 
                                                    random_state=532, shuffle=True, stratify = Y)
Xcv = X[CV_ind]
Xval = X[val_ind]

Xval_transformed = zscore(Xval, axis=1)
if nn_choice in [1, 2, 4]:
    Xval_transformed = Xval_transformed[...,None]
Yval_OH = to_categorical(Yval+1, 3)

# 5 -fold cv 
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=321)
indexes = skf.split(Xcv, Ycv)


fold_count = 0 
summary = []
summary_weighted = []
ConM = []
for train_index, test_index in indexes:
    print('Fold {} started.'.format(fold_count))
    
    Xtrain, Xtest = Xcv[train_index], Xcv[test_index]
    Ytrain, Ytest = Ycv[train_index], Ycv[test_index]
    
    '''
    Normalize
    '''
    X_train_transformed = zscore(Xtrain, axis=1)
    X_test_transformed = zscore(Xtest, axis=1)
    
    if nn_choice in [1, 2]:
        X_train_transformed = X_train_transformed[...,None]
        X_test_transformed = X_test_transformed[...,None]
    
    Ytrain_OH = to_categorical(Ytrain+1, 3)
    Ytest_OH = to_categorical(Ytest+1, 3)
    
    #%% Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=10, min_lr=1e-4)

    cpt_path = os.path.join(path, 'ckpt/S{}_checkpoint_{}_{}chns_fold{}'.format(subject, nn_token, chns, fold_count))
    # print(cpt_path)
        
    cpt = ModelCheckpoint(filepath=cpt_path,
                          save_weights_only=True,
                          monitor='val_accuracy',
                          mode='max',
                          save_best_only=True)
    
    #%% Training
    model.load_weights('model_ini.h5') # each fold starting with the same initialization
    model.fit(X_train_transformed, Ytrain_OH, 
              epochs = epochs, batch_size = batch_size,
              # validation_split=0.3,
              validation_data = (Xval_transformed, Yval_OH),
              verbose=1,
              callbacks=[reduce_lr, cpt],
              shuffle = True
             )

    model.load_weights(cpt_path)
    pred = model.predict(X_test_transformed)
    CM = confusion_matrix(Ytest+1, np.argmax(pred, axis=1))
    # print(CM)
    
    a, b = scores(CM )
    # print(b)
    summary.append(a)
    summary_weighted.append(b)
    ConM.append( CM )
    
    
    print('Fold {} finished.'.format(fold_count))   
    print('#'*40)
    fold_count += 1

summary = np.array(summary)
summary_weighted = np.array(summary_weighted)

# print('mean: {}'.format(np.mean(summary, axis = 0)))
# print('std: {}'.format(np.std(summary, axis = 0))) 

with open('./exp_history/exp_{}_history.txt'.format(nn_token), 'a') as file:
    file.write('\n' + '='*60 + '\n')
    file.write('{}\'s performance on Subject {} with {}: \n'.format(nn_token, subject, chns))
    # file.write('mean:   ')
    # file.writelines(['{:.04f}    '.format(s) for s in np.mean(summary, axis = 0).reshape(-1)])
    # file.write('\n')
    # file.write('std:  ')
    # file.writelines(['{:.04f}    '.format(s) for s in np.std(summary, axis = 0).reshape(-1)])
    file.write('\n')
    file.write('mean(W):  ')
    file.writelines(['{:.04f}    '.format(s) for s in np.mean(summary_weighted, axis = 0).reshape(-1)])
    file.write('\n')
    file.write('std(W):  ')
    file.writelines(['{:.04f}    '.format(s) for s in np.std(summary_weighted, axis = 0).reshape(-1)])
    file.write('\n')

    
np.save(os.path.join(path, 'benchmark_summary/S{}_{}_{}chns'.format(subject, nn_token, chns)), summary)
np.save(os.path.join(path, 'benchmark_summary/SW{}_{}_{}chns'.format(subject, nn_token, chns)), summary_weighted)
np.save(os.path.join(path, 'benchmark_summary/CM_S{}_{}_{}chns'.format(subject, nn_token, chns)), ConM)
# total_CM = ConM[0] + ConM[1] + ConM[2]
# plot_confusion_matrix(total_CM, ['Negative', 'Neutral', 'Positive'], True)
# %%