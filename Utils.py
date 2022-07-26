# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:21:25 2020

@author: dykua

"""
#%%
import numpy as np
from scipy import signal
import os

def band_decompose(X, fs=400, bands={'Delta': (0, 4),
                                     'Theta': (4, 8),
                                     'Alpha': (8, 12),
                                     'Beta': (12, 30),
                                     'Gamma': (30, 45)}):
    '''
    extract relative band energy from signal `X` according to `bands`
    X: (batchsize, samples, channels)
    fs: sampling frequency
    bands: a dict of frequecy bands
    
    ? is the following correct ?
    Delta wave – (0.5 – 3 Hz)
    Theta wave – (4 – 7 Hz)
    Alpha wave – (7 – 15 Hz)
    Mu wave – (7.5 – 12.5 Hz)
    SMR wave – (12.5 – 15.5 Hz)
    Beta wave – (15 – 30 Hz)
    Gamma wave – (>30 Hz)
    '''
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(X, axis = 1))
    
    total_energy = np.sum(fft_vals**2, axis=1)
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(X.shape[1], 1.0/fs)
    band_power = []
    for band in bands:  
        freq_ix = np.where((fft_freq >= bands[band][0]) & 
                           (fft_freq <= bands[band][1]))[0]
        
#        band_power.append(np.median(fft_vals[:,freq_ix,:], axis = 1)) # what metric to use here?
        band_power.append(np.sum(fft_vals[:,freq_ix,:]**2, axis = 1)/total_energy)
    
    return np.stack(band_power, axis=1) # normalize?


'''
bandpass filter, copied from 
https://users.soe.ucsc.edu/~karplus/bme51/w17/bandpass-filter.py
''' 
   
def band_pass(values, low_end_cutoff, high_end_cutoff, sampling_freq, btype = 'bandpass'):
    # The band-pass filter will pass signals with frequencies between
    # low_end_cutoff and high_end_cutoff
    lo_end_over_Nyquist = low_end_cutoff/(0.5*sampling_freq)
    hi_end_over_Nyquist = high_end_cutoff/(0.5*sampling_freq)
    
    # If the bandpass filter gets ridiculously large output values (1E6 or more),
    # the problem is numerical instability of the filter (probably from using a
    # high sampling rate).  
    # The problem can be addressed by reducing the order of the filter (first argument) from 5 to 2.
    bess_b,bess_a = signal.iirfilter(5,
                Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],
                btype=btype, ftype='bessel')
    bandpass = signal.filtfilt(bess_b,bess_a,values)
    
# =============================================================================
#     # The low-pass filter will pass signals with frequencies
#     # below low_end_cutoff
#     bess_b,bess_a = scipy.signal.iirfilter(5, Wn=[lo_end_over_Nyquist],
#                 btype="lowpass", ftype='bessel')
#     lowpass = scipy.signal.filtfilt(bess_b,bess_a,values)
# =============================================================================
    
    return bandpass

def batch_band_pass(values, low_end_cutoff, high_end_cutoff, sampling_freq, btype='bandpass'):
    assert len(values.shape) == 3, "wrong input shape"
    S, T, C = values.shape
    X_filtered = np.empty(values.shape)
    lo_end_over_Nyquist = low_end_cutoff/(0.5*sampling_freq)
    hi_end_over_Nyquist = high_end_cutoff/(0.5*sampling_freq)

    bess_b,bess_a = signal.iirfilter(5,
                Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],
                btype=btype, ftype='bessel')
                
    for i in range(S):
        for j in range(C):
            X_filtered[i,:,j] = signal.filtfilt(bess_b,bess_a,values[i,:,j])
    
    return X_filtered

def check_summary(st_path, subject, chns, model_token):
    for m in model_token:
        temp = np.load(os.path.join(st_path, 'S{:02d}_{}_{}chns.npy'.format(subject, m, chns) ) )
        print('Model {}: '.format(m))
        print(temp)
        print(np.mean(temp, axis=0))

def find_idx_for_iprm(st_path, chns, comp_pair):
    re = []
    for i in range(1, 16):
        A_p = np.load(os.path.join(st_path, 'S{:02d}_{}_{}chns.npy'.format(i, comp_pair[0], chns) ) )
        B_p = np.load(os.path.join(st_path, 'S{:02d}_{}_{}chns.npy'.format(i, comp_pair[1], chns) ) )
        re.append(np.mean(B_p,0)[0] - np.mean(A_p, 0)[0])
    return re

def scores(CM):
    '''
    Calculate the accuracy, precision, recall, specificity and f1_score from confusion matrix
    It returns a matrix with class labels as rows and scores above as columns
    '''
    n_classes = CM.shape[0]
    scores = np.empty((n_classes, 5))
    _sum = np.sum(CM)
    col_sum = np.sum(CM, axis=0)
    row_sum = np.sum(CM, axis=1)
    TP = CM.diagonal()
    scores[:,0] = TP/_sum # accuracy per class, sum up them for accuracy
    scores[:,1] = TP/col_sum # precision per class
    scores[:,2] = TP/row_sum  # recall per class 
    scores[:,3] = (_sum-row_sum-col_sum+TP)/(_sum-row_sum)
    scores[:,-1] = 2*(scores[:,1]*scores[:,2])/(scores[:,1]+scores[:,2]) # F1_score
    
    weight = row_sum/_sum
    weighted_summary = np.array([np.sum(scores[:,0]),
                                 np.dot(weight, scores[:,1]), 
                                 np.dot(weight, scores[:,2]),
                                 np.dot(weight, scores[:,3]),
                                 np.dot(weight, scores[:,-1])])
#    print(weight)
    return scores, weighted_summary


# if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # fs= 400
    # dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
    # subject = 1
    # Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
    # # Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
    # # Xtest = np.load(dataset+r'S{}test.npy'.format(subject))
    # # Ytrain = np.load(dataset+r'Ytrain.npy'.format(subject)) # ordering are the same for both subjects: 40 + 40 + 40+ 40
    # # Ytest = np.load(dataset+r'S{}Ytest.npy'.format(subject))[0] 
    # # Xtrain = np.load(dataset+r'S{}train_gen.npy'.format(subject))
    
    # # bands = [[0.5, 3], [4,7], [7, 15], [15, 30], [30, fs//2-0.5]] # Delta, theta, alpha, beta, gamma
    # bands = [[0.5,5]] + [[5*i, 5*i+5] for i in range(1, 10)] + [[50, fs//2-0.5]]
    # stacked = []
    # for band in bands:
    #     stacked.append( batch_band_pass(Xtrain, band[0], band[1], fs))
    
    # sample= 0 
    # channel = 0
    # plt.subplot(len(bands)+1,1,1)  
    # plt.plot(Xtrain[sample,:,channel])
    # for i in range(len(bands)):
    #     plt.subplot(len(bands)+1,1,i+2)
    #     plt.plot(stacked[i][sample,:,channel])
        
        
    
        
        
        
    
    
    