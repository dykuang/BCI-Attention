'''
Extra plots for SEED data with the MCAM module
'''
'''
Make candidate plots and summaries from DEAP experiments
'''

#%%
import tensorflow as tf
# tf.compat.v1.enable_eager_execution() 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix
from Utils import scores
from visual import plot_confusion_matrix
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

#%%
'''
Plots about trained models
'''
from Models import *

def load_trained(ckpt_path, nn_token, subject,model_dict, count = 0, num_class=3, 
                 seg_len=200, lr=1e-2):
    if nn_token in ['baseline', 'eegnet']:
        model = EEGNet(nb_classes = num_class, Chans = 62, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    learning_rate = lr)
    elif nn_token == 'SE':
        model = CANet(nb_classes= num_class, Chans = 62, Samples = seg_len, attention_module = 'se_block',
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token =='CBAM':
        model = CANet(nb_classes = num_class, Chans = 62, Samples = seg_len, attention_module='cbam_block',
                    dropoutRate = 0.5, kernLength = 5, F1 = 8,
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token in ['C2A_NNR_ID', 'C2A_NNR_mono_ID']:
        model = MTNet(nb_classes= num_class, Chans = 62, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    penalty_rate= 0.1, mono_mode ='ID',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token in ['C2A_NNR_DI', 'C2A_NNR_mono_DI']:
        model = MTNet(nb_classes= num_class, Chans = 62, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    penalty_rate= 0.1, mono_mode ='DI',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token == 'C2A_NNR_0c':
        model = MTNet(nb_classes = num_class, Chans = 62, Samples = seg_len,
                    dropoutRate = 0.5, kernLength = 5, F1 = 8,
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    penalty_rate = 0.0, mono_mode = None,
                    optimizer = Adam, learning_rate = lr) 
    elif nn_token == 'qkv':
        model = QKVNet(nb_classes = num_class, Chans = 62, Samples = seg_len, 
                  dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                  D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                  optimizer = Adam, learning_rate = lr)     
    elif nn_token in ['K_v1', 'kanet_v1']:
        model = KANet(nb_classes = num_class, Chans = 62, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    optimizer = Adam, learning_rate = lr)  
    elif nn_token in ['DCN']:
        model = DeepConvNet(nb_classes = num_class, Chans = 62, Samples = seg_len,
                    dropoutRate = 0.25, attention_type = 'No',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token in ['DCN_KAM']:
        model = DeepConvNet(nb_classes = num_class, Chans = 62, Samples = seg_len,
                    dropoutRate = 0.25, attention_type = 'KAM',
                    optimizer = Adam, learning_rate = lr)
    else:
        assert 'nn_token not recognized.'
    
    try:
        model.load_weights(os.path.join(ckpt_path, 'S{:02d}_checkpoint_{}_62chns_fold{}'.format(subject,model_dict[nn_token],count)))
    except KeyError:
        model.load_weights(os.path.join(ckpt_path, 'S{:02d}_checkpoint_{}_62chns_fold{}'.format(subject,nn_token,count)))

    return model

#
#%%
'''
Extracting channel attention weights for compared models:
'''
ckpt_path = '/mnt/HDD/Datasets/SEED/ckpt'
subject_selected = 1
model_names = ['baseline','qkv','SE','CBAM','C2A_NNR_0c','C2A_NNR_ID','C2A_NNR_DI']
model_tokens = ['eegnet', 'qkv','SE','CBAM','C2A_NNR_0c','C2A_NNR_mono_ID','C2A_NNR_mono_DI']
# model_names = ['baseline','qkv','SE','CBAM','K_v1']
# model_tokens = ['eegnet', 'qkv','SE','CBAM','kanet_v1']
# model_names = ['baseline','qkv','SE','CBAM','C2A_NNR_DI','C2A_NNR_ID','C2A_NNR_0c', 'K_v1']
# model_tokens = ['eegnet', 'qkv','SE','CBAM','C2A_NNR_mono_DI','C2A_NNR_mono_ID','C2A_NNR_0c','kanet_v1']
model_dict = dict(zip(model_names, model_tokens))

#%%
'''
Extracting kernel weights in the depthwise conv layer for visualizing wiht scalp maps
'''

# for _s in [15]:
#     Collect = []
#     for nn_token in model_names:
#         W_list = []
#         for fld  in range(5):
#             model = load_trained(os.path.join(ckpt_path,nn_token), nn_token, _s, count = fld, 
#                                 num_class=3, seg_len=200, lr=1e-3,model_dict=model_dict)
#             # model = load_trained(ckpt_path, nn_token, subject_selected, count = fld, 
#             #                      num_class=3, seg_len=200, lr=1e-3,model_dict=model_dict)
#             W = model.get_layer('DepthConv').weights
#             W_list.append(W[0][0].numpy())
#         W_list = np.concatenate(W_list, axis=-1)
#         Collect.append(W_list)


#     ## Create target array to save, after normalization

#     CC = np.array(Collect)[...,0,::2]
#     savemat('/mnt/HDD/Datasets/SEED/benchmark_summary/ATT_SEED_7models_S{}.mat'.format(_s), {'CM':CC})

for _s in [1, 4, 9, 14, 15]:
    Collect = []
    for nn_token in model_names:
        W_list = []
        for fld  in range(5):
            model = load_trained(os.path.join(ckpt_path,nn_token), nn_token, _s, count = fld, 
                                num_class=3, seg_len=200, lr=1e-3,model_dict=model_dict)
            # model = load_trained(ckpt_path, nn_token, subject_selected, count = fld, 
            #                      num_class=3, seg_len=200, lr=1e-3,model_dict=model_dict)
            W = model.get_layer('DepthConv').weights 
            W_list.append(W[0][0].numpy()) # (62, 8, 2)
        W_list = np.concatenate(W_list, axis=-1) # (62, 8, 10)
        Collect.append(W_list.reshape(62,-1))


    ## Create target array to save, after normalization

    CC = np.array(Collect)
    # savemat('/mnt/HDD/Datasets/SEED/benchmark_summary/SEED_scalp_all_models_S{:02d}.mat'.format(_s), {'CM':CC})


#%%
'''
Try exploring the weight correlation among sensor locations
'''
from scipy.io import loadmat
import pickle

# For each model, get the most correlated channel and max correlated channel pair

def get_most_correlated_pairs(CMatrix, topN=5, symmetry=True):
    '''
    Assuming Cmatrix is 2d
    '''
    rr, cc = CMatrix.shape
    if symmetry:
        coord_2d = []
        val = []
        for i in range(1, rr):
            for j in range(i):
                coord_2d.append((i,j))
                val.append(CMatrix[i,j])
        sorted = np.argsort(np.array(val))
        max_loc = [coord_2d[ii] for ii in sorted[-topN:]]
        min_loc = [coord_2d[ii] for ii in sorted[:topN]]

        
    else:
        sorted = np.argsort(np.reshape(CMatrix, -1))
        
        max_cor_idx = sorted[-topN:]
        min_cor_idx = sorted[:topN]
        
        max_loc = [(idx//rr, idx%cc) for idx in max_cor_idx]
        min_loc = [(idx//rr, idx%cc) for idx in min_cor_idx]

    
    return max_loc, min_loc


def link_cor_2d(Cmatrix, Coord_2d, topN=5):

    max_loc, min_loc = get_most_correlated_pairs(Cmatrix, topN)

    plt.figure(figsize=(6,6))
    plt.plot(Coord_2d[:,0], Coord_2d[:,1], 'k.', markersize=20)
    for _p in max_loc:
        plt.plot([Coord_2d[_p[0],0],  Coord_2d[_p[1],0]],
                [Coord_2d[_p[0],1],  Coord_2d[_p[1],1]], 'r', linewidth=3)
    for _p in min_loc:
        plt.plot([Coord_2d[_p[0],0],  Coord_2d[_p[1],0]],
                [Coord_2d[_p[0],1],  Coord_2d[_p[1],1]], 'b',linewidth=3)

    plt.xlim([-0.6,0.6])
    plt.ylim([-0.6, 0.6])
    plt.grid(False)
    plt.axis('off')

#%%

scalp_weights = loadmat('/mnt/HDD/Datasets/SEED/benchmark_summary/SEED_scalp_all_models_S{:02d}.mat'.format(subject_selected))['CM']

with open('./ch_pos_1020.pkl', 'rb') as pkl:
    pos_dict = pickle.load(pkl)

XY = []
for ch in pos_dict.keys():
        XY.append(pos_dict[ch][:2])
XY = np.array(XY)


for _w in scalp_weights:
    cor = np.corrcoef(_w)
    link_cor_2d(cor, XY)

#%%
'''
Get performance metrics
'''
summary_path = '/mnt/HDD/Datasets/SEED/benchmark_summary'
summary_dict = dict.fromkeys(model_tokens)
subjects = [ i for i in range(1,16) ]

for _folder, _token in model_dict.items():
    temp_dict = dict.fromkeys(subjects)
    for _s in subjects:
        S_temp = np.load(os.path.join(summary_path, _folder, 'S{:02d}_{}_62chns.npy'.format(_s, _token) ) )
        if _folder == 'baseline':
            SW_temp = np.load(os.path.join(summary_path, _folder, 'S{:02d}_{}_62chns.npy'.format(_s, _token) ) )
        else:
            SW_temp = np.load(os.path.join(summary_path, _folder, 'SW{:02d}_{}_62chns.npy'.format(_s, _token) ) )
        CM_temp = np.load(os.path.join(summary_path, _folder, 'CM_S{:02d}_{}_62chns.npy'.format(_s, _token) ) )
        temp_dict[_s] = {'Score': S_temp,
                         'Weighted_Score': SW_temp,
                         'CM': CM_temp
                         }
    
    summary_dict[_token] = temp_dict


#%%
def get_perform_overall(model_name):
    ccc = []
    for s in subjects:
        ccc.append( summary_dict[model_name][s]['Weighted_Score'])
    ccc = np.array(ccc)    #accuracy, precision, recall, specificity and f1_score
    m_per_sub = np.mean(ccc, axis=1)
    std_per_sub = np.std(ccc, axis=1)
    m_all = np.mean(m_per_sub, axis=0)
    std_all = np.std(m_per_sub, axis=0)

    return m_per_sub, std_per_sub, m_all, std_all

from scipy.stats import ttest_rel
P_EEG  = get_perform_overall('eegnet')
P_CBAM = get_perform_overall('CBAM')
P_SE = get_perform_overall('SE')
P_M0 = get_perform_overall('C2A_NNR_0c')
P_M_ID = get_perform_overall('C2A_NNR_mono_ID')
P_M_DI = get_perform_overall('C2A_NNR_mono_DI')
P_QKV  = get_perform_overall('qkv')

F1 = []
F1_std = []

for _m in model_tokens:
    temp_m = get_perform_overall(_m)[0][:,-1]
    temp_std= get_perform_overall(_m)[1][:,-1]
    F1.append(temp_m)
    F1_std.append(temp_std)

F1 = np.array(F1)
F1_std = np.array(F1_std)
best_model_per_sub = np.argmax(F1, axis=0)

print([model_names[i] for i in best_model_per_sub])


# ttest_rel(P_EEG[0][:,-1], P_M_DI[0][:,-1], alternative='less') 
ttest_rel(P_EEG[0][:,-1], P_M_ID[0][:,-1], alternative='less') 
ttest_rel(P_SE[0][:,-1], P_M_ID[0][:,-1], alternative='less') 
ttest_rel(P_CBAM[0][:,-1], P_M_ID[0][:,-1], alternative='less') 
ttest_rel(P_QKV[0][:,-1], P_M_ID[0][:,-1], alternative='less') 
# ttest_rel(P_M0[0][:,-1], P_M_ID[0][:,-1], alternative='less') 
# %%
#==============================================================================
# A grouped boxplot
#==============================================================================
import seaborn as sns
import pandas as pd

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color, facecolor = color, linewidth=2)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
#    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=2)
    plt.setp(bp['fliers'], markersize=4)

def show_box_group(data, names, ticks, colors, box_width = 0.3, sparsity = 3, ymin=0, ymax = 1,
                   style = 'bmh'):
    # with plt.style.context(style):
        plt.figure()
        for i, sample in enumerate(data):
            bp = plt.boxplot(sample, positions=np.array(np.arange(sample.shape[1]))*sparsity-0.6+0.4*i,  
                    widths=box_width, sym = 'o',
                    notch=True, patch_artist=True)
            set_box_color(bp, colors[i])
            for patch in bp['boxes']:
                patch.set_alpha(0.8)
            plt.plot([], c=colors[i], label=names[i])
        plt.legend(loc='upper right')

        plt.xticks(np.arange(0, len(ticks) * sparsity, sparsity), ticks, rotation = 45)
        plt.xlim(-2, len(ticks)*sparsity-0.4)
        plt.ylim(ymin, ymax)
        # plt.ylabel('Dice Score')
        #plt.title('Different methods on selected regions')
        plt.grid()
        plt.tight_layout()

ticks = ['Acc.', 'Prec.', 'Spec.', 'F1']
# colors = ['#2C7BB6', '#999900', '#2ca25f', '#9400d3','#636363']
palette = sns.color_palette("tab10", 10).as_hex()
colors = palette[:5]
box_width = 0.3
sparsity = 3 

summary2plot = [ P_EEG[0][:,[0, 1, 3, 4]], 
                 P_QKV[0][:, [0, 1, 3, 4]], 
                 P_SE[0][:, [0, 1, 3, 4]], 
                 P_CBAM[0][:, [0, 1, 3, 4]], 
                 P_M_ID[0][:, [0, 1, 3, 4]]
]
legend_list = ['EEGNet', '+QKV', '+SE', '+CBAM', '+M2']
show_box_group(summary2plot , legend_list,  ticks, colors, ymin=0.8, ymax=1.0)

# %%
'''
getting best model across fold
'''

def make_segs(data, seg_len, stride):
    t_len = data.shape[1]
    segs = np.stack([data[:,i*stride:i*stride+seg_len,:] for i in range(t_len//stride) if i*stride+seg_len<=t_len], axis= 1)
    # print(segs.shape)
    return segs.reshape((-1, seg_len, data.shape[-1]))

def get_best_model(ckpt_path, nn_token, subject_selected):
    score = summary_dict[nn_token][subject_selected]['Weighted_Score']
    fld = np.argmax(score[:,0]) # select a metric
    model = load_trained(ckpt_path, nn_token, subject_selected, 
                         model_dict, count = fld, num_class=3, seg_len=200, lr=1e-3)
    return model  
#%%
'''
start here for a different subject selected
'''
data_path = '/mnt/HDD/Datasets/SEED'
ckpt_path = '/mnt/HDD/Datasets/SEED/ckpt'
cpr_model={}
for _m,_token in model_dict.items():
    cpr_model[_m] = get_best_model(os.path.join(ckpt_path, _m),
                                   _token, subject_selected) 

#%%
'''
check monotonicity
'''
pts = np.linspace(-1.0, 1.0, 21)
def get_att_map(model):

    x_In = layers.Input((1,1,1), name='map_in')
    x = model.get_layer('att_mono').D1(x_In)
    x= model.get_layer('att_mono').D2(x)
    x= model.get_layer('att_mono').BN(x)
    x_Out= model.get_layer('att_mono').Pred(x)

    return Model(x_In, x_Out)

mono_list = []
for folder_token, nn_token in zip(['C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI'], ['C2A_NNR_0c', 'C2A_NNR_mono_ID', 'C2A_NNR_mono_DI']):
    temp_list = []
    for fld  in range(5):
        model = load_trained(os.path.join(ckpt_path, folder_token), nn_token, subject_selected, 
                             model_dict, count = fld, num_class=3, seg_len=200, lr=1e-3)
        att = get_att_map(model)
        cc = att.predict(pts.reshape([-1,1,1,1]))[:,0,0,0]
        temp_list.append(cc)
    mono_list.append(np.array(temp_list))

with plt.style.context('ggplot'): # compare with results from SEED
    fig, ax = plt.subplots(1,3,figsize=(12,6))
    for i in range(3):
        _mean = np.mean(mono_list[i], axis=0)
        _std = np.std(mono_list[i], axis=0)
        ax[i].plot(pts, _mean, 'b-', label='mean')
        ax[i].fill_between(pts, _mean - _std, _mean + _std, color='pink', alpha=0.8, label='std')        
        ax[i].axvline(x=0, color='red', linestyle='--')
        ax[i].set_ylim([0, 1])  
        ax[i].set_title('M{}'.format(i+1))  
        ax[i].legend(loc = 'upper left')  

#%%
'''
Get some sample feature slices
'''
X = loadmat( os.path.join(data_path, 'S{:02d}_E01.mat'.format(subject_selected)) )['segs'].transpose([2,1,0])
chns = np.arange(62)
X_normalized = zscore(X, axis=1)
chns_token = '{:02d}'.format(len(chns))
Y = loadmat( os.path.join(data_path, 'Label.mat') )['seg_labels'][0]+1

#%%
'''make some sample feature plots'''
feature_list = []
for folder_token, nn_token in zip(['C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI'], 
                                  ['C2A_NNR_0c', 'C2A_NNR_mono_ID', 'C2A_NNR_mono_DI']):
    temp = []
    for fld  in range(5):
        model = load_trained(os.path.join(ckpt_path, folder_token), nn_token, subject_selected, 
                             model_dict, count = fld, num_class=3, seg_len=200, lr=1e-3)
        fmap = Model(model.input,  
                     model.get_layer('att_mono').output[0] 
                     - model.get_layer('SepConv-1').output[...,0,:] ) #get the difference
        temp.append(fmap.predict(X_normalized))

    feature_list.append(np.array(temp))

feature_list = np.array(feature_list) # (monotype, folds, samples, ...)
feature_list_m = np.mean(feature_list, axis= 1) # mean over folds

rnd_ind = 10

#%%
Feature2Plot = feature_list_m[:,[np.where(Y==i)[0][rnd_ind] for i in range(3)],...]  #(monotype, label, ...)
# Feature2Plot = Feature2Plot/np.max(np.abs(Feature2Plot), keepdims=True) # normalization 

#%%
with plt.style.context('ggplot'): # compare with results from SEED
    fig, ax = plt.subplots(3,3,figsize=(8,6))
    for i in range(3):
        for j in range(3):
            _im = ax[i][j].imshow(Feature2Plot[j,i].T, cmap = 'jet')
            ax[i][j].set_axis_off()
    plt.colorbar(_im, ax=ax.ravel().tolist() )         

# %%
'''
Sensitivity on Frequencies
'''
from scipy import signal
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




#%%
# data_seq = [ batch_band_pass(X, 0.1, hp, 200) for hp in [10*i for i in range(1,10)]] #bandpass case

#=================
rej_band = [(l, l+10) for l in range(10,90,10)]
rej_band.append((90, 99.99))
rej_band.insert(0, (1,10))

data_seq = [ batch_band_pass(X, lp, hp, 200, btype='bandstop') for 
             (lp,hp) in rej_band ] #bandstop case
data_seq.append(X)
data_seq = [zscore(_d, axis=1) for _d in data_seq] # normalization after filtering

# X_N = zscore(X, axis=1)
# data_seq = [ batch_band_pass(X_N, lp, hp, 200, btype='bandstop') for 
#              (lp,hp) in rej_band ] #bandstop case
# data_seq.append(X_N)
#=====================

#%%
freq_dict = {}

for _name, _m in cpr_model.items():
    freq_p = []
    for _data in data_seq:
        pred = _m.predict(_data)
        # CM = confusion_matrix( baseline_label, np.argmax(pred, axis=1))
        CM = confusion_matrix( Y, np.argmax(pred, axis=1) )
        # print(CM)
        
        _, b = scores(CM )
        freq_p.append(b )
    freq_dict[_name] = np.array(freq_p)   

#======================
# band pass case
# plt.figure()
# freq_grid = [10*i for i in range(1,11)]
# ll = ['EEGNet', '+QKV',  '+SE', '+CBAM', '+KAM']
# ll = ['EEGNet', '+QKV',   '+CBAM', '+SE','+M1', '+M2', '+M3']
# count = 0
# for _k, _v in freq_dict.items():
#     plt.plot(freq_grid, _v[:,0], 'd--',label = ll[count])
#     count += 1
# plt.ylim([0.25, 1.0])
# plt.xlabel('Hz')
# plt.ylabel('Acc')
# plt.legend(loc = 'upper left')

# for _k in ['baseline', 'qkv', 'CBAM', 'SE', 'C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI']:
#     plt.plot(freq_dict[_k][:,0], 'd--',label = ll[count])
#     count += 1
# plt.ylim([0.25, 1.0])
# plt.xlabel('Hz')
# plt.ylabel('Acc')
# plt.legend(loc = 'upper left')
# plt.grid()
#===========================================

#%%
##================================
# For bandreject case
# with plt.style.context('ggplot'):    
plt.figure()
freq_grid = [10*i for i in range(1,11)]
ll = ['EEGNet', '+QKV',  '+SE', '+CBAM', '+KAM']
# ll = ['EEGNet', '+QKV',   '+CBAM', '+SE','+M1', '+M2', '+M3']
count = 0
# for _k in ['baseline', 'qkv', 'CBAM', 'SE', 'C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI']:
for _k in ['baseline', 'qkv', 'CBAM', 'SE', 'K_v1']:
    plt.plot(freq_dict[_k][:,0], 'd--',label = ll[count])
    count += 1
plt.ylim([0.5, 1.0])
plt.xticks(np.arange(freq_dict[_k].shape[0]), 
        labels=['1-10', '10-20', '20-30', '30-40', '40-50', '50-60',
                '60-70', '70-80', '80-90', '90-99.99', 'original'], rotation=-45)
plt.legend(loc = 'lower left')
plt.xlabel('Hz')
plt.ylabel('Acc')
plt.grid()
##=================================
#%%
'''
Prediction Transition Curve
'''

'''
Finding those segments where all models predicted correctly
'''
def get_CM_idx(model, X, Y):
    num_cls = len(np.unique(Y))
    # prepare samples
    seg_pred_score = model.predict(X)
    seg_pred_label = np.argmax(seg_pred_score, axis=-1)

    idx2vis_correct = []
    idx2vis_wrong = []
    for i in range(num_cls):
        idx2vis_correct.append( np.where( np.logical_and( seg_pred_label == Y, seg_pred_label==i))[0] )
        idx2vis_wrong.append( np.where( np.logical_and( seg_pred_label != Y, seg_pred_label==i))[0]  )
    
    return idx2vis_correct, idx2vis_wrong

Correct_list = []
X_normalized = zscore(X, axis=1)
for _name, _m in cpr_model.items():
    temp_correct, _ = get_CM_idx(_m, X_normalized, Y)
    Correct_list.append(temp_correct)

def common(lst1, lst2): 
    return list(set(lst1) & set(lst2))

Common_idx = []
for j in range(3): #j for labels
    a = Correct_list[0][j].copy()
    for i in range(1,len(Correct_list)): #i for models
        a = common(a, Correct_list[i][j])
    Common_idx.append(a)

for i in range(3):
    print(len(Common_idx[i]))

#%%
'''
Morphing between curves belonging to different labels
'''
def morphed_curve(cA, cB, grid=[0, 0.25, 0.5, 0.75, 1.0]):
    track = []
    for _h in grid:
        track.append((1-_h)*cA + _h*cB)
    return track

ind_select = 0
segs_for_morph = np.array(X_normalized[[c[ind_select ] for c in Common_idx]])


from itertools import combinations

interp_grid = np.linspace(0,1,101)
model_compared = list(cpr_model.keys())
trackDict = dict.fromkeys(model_compared)
crossDict = dict.fromkeys(model_compared)
for _k in trackDict.keys():
    temp_track = []
    temp_cross = []
    # i = model_compared.index(_k)
    for _p in combinations(np.arange(len(segs_for_morph)), 2):      
        track = morphed_curve(segs_for_morph[_p[0]], segs_for_morph[_p[1]], interp_grid)
        temp_track.append( cpr_model[_k].predict(np.array(track)[...,None]) )
        temp_cross.append( (interp_grid[1]-interp_grid[0])*np.argmin( abs(temp_track[-1][:,_p[0]] - temp_track[-1][:,_p[1]]) )  )
    trackDict[_k] = temp_track
    crossDict[_k] = temp_cross

# for i in trackDict['eegnet']:
#     plt.figure()
#     plt.plot(interp_grid, i, '+--')
#     plt.legend(['Neg', 'Neu', 'Pos'])


#%%
# def trig_abl_plot(cross_track):
#     theta = np.array([90, 210, 330, 90])
#     theta = theta*np.pi/180.0
#     Data2Plot = [ [ cross_track[0], 1, cross_track[1]],  
#                   [1, 1 - cross_track[0],  cross_track[2]],
#                   [ 1 - cross_track[1], 1-cross_track[0], 1]
#                  ]
#     labels = ['Neu.', 'Neg.', 'Pos.'] #(1, 0, 2)
#     fig_label = ['Neg.', 'Neu.', 'Pos.']
    
#     colors = ['g', 'orange', 'cyan']
#     plt.figure()
#     for i in range(3):
#         ax = plt.subplot(1,3,i+1, projection='polar')
#         temp = np.append(Data2Plot[i], Data2Plot[i][0])
#         ax.plot(theta, temp,colors[i])
#         ax.fill(theta, temp,colors[i], alpha=0.3)
#         ax.set_xticks(theta[:3])
#         ax.set_xticklabels(labels, y=0.1)
#         ax.set_ylim([0, 1.0])
#         ax.set_title('Origin-{} \n {:.02f}'.format(fig_label[i], np.sum(temp[:3])), color=colors[i], size=10)


# trig_abl_plot(crossDict['eegnet'])
# %%
'''
make trajactory plots on the hyperplane x+y+z = 1
'''
def trace_plot_on_hp(cross_track, attach_legend=True, title=None):
    theta = np.array([90, 210, 330, 90])
    theta = theta*np.pi/180.0
    
    center = np.array([1/3, 1/3, 1/3])
    vec_x = np.array([-0.5**0.5, 0.5**0.5, 0])
    vec_y = np.array([-1/3, -1/3, 2/3])
    ratio = np.linalg.norm(vec_y)
    vec_y = vec_y/ ratio

    labels = ['Neu.', 'Neg.', 'Pos.'] #(1, 0, 2)
    
    # colors = ['g', 'orange', 'cyan']
    colors = ['r', 'g', 'b']
    legend = ['Neg-Neu', 'Neg-Pos', 'Neu-Pos']
    plt.figure()
    ax = plt.subplot(1,1,1, projection='polar')
    for i in range(3):       
        coord = np.array([cross_track[i][:,0], 
                          cross_track[i][:,2], 
                          cross_track[i][:,1]]).transpose() - center  #neg-x, pos-y, neu-z
        # print(coord.shape)
        proj_x = coord @ vec_x[...,None]
        proj_y = coord @ vec_y[...,None]
        phi = np.arctan2(proj_y, proj_x) 
        phi = np.where(phi>0, phi, phi+np.pi*2)
        r = np.linalg.norm(coord, axis=1)
        ax.plot(phi, r/ratio, colors[i], linewidth=6, label=legend[i])

    ax.plot(theta, [1 for i in range(4)], 'k--')
    # ax.fill(theta, np.array([]),colors[i], alpha=0.3)
    ax.set_xticks(theta[:3])
    ax.set_xticklabels(labels, y=0, size=12)
    ax.set_ylim([0, 1.0])
    if attach_legend:
        ax.legend(loc='upper right')
    ax.set_title(title, loc='left')

trace_plot_on_hp(trackDict['baseline'], attach_legend=False, title = 'EEGNet')
# trace_plot_on_hp(trackDict['SE'], attach_legend=False, title = '+SE')
# trace_plot_on_hp(trackDict['CBAM'], attach_legend=True, title = '+CBAM')
# %%
'''
Extraplot for tracking the accuracy change while morphing from zeros to X linearly on the amplitude
'''

from sklearn.metrics import accuracy_score
interp_grid = np.linspace(0,1,21)
AccTrackDict = dict.fromkeys(cpr_model.keys())
count_Neg = np.sum(Y==0)  #Negative
count_Neu = np.sum(Y==1)  #Neutral
count_Pos = np.sum(Y==2)  #Positive
count_all = Y.size

for _k in AccTrackDict.keys():
    temp_track = []
    for alpha in interp_grid:
        pred = cpr_model[_k].predict(alpha*X_normalized)
        pred_int = np.argmax(pred,axis=1)
        CM = confusion_matrix( Y , pred_int )

        temp_track.append( [CM[0,0]/count_Neg, CM[1,1]/count_Neu,
                            CM[2,2]/count_Pos,
                            accuracy_score(Y, pred_int)
                            ] )
        temp_track[-1] += [(3/2)**0.5*(count_Neg/count_all*(temp_track[-1][0]-temp_track[-1][-1])**2
                          + count_Neu/count_all*(temp_track[-1][1]-temp_track[-1][-1])**2
                          + count_Pos/count_all*(temp_track[-1][2]-temp_track[-1][-1])**2)**0.5]#append the weighted std

        # temp_track.append( [np.sum(pred_int==0)/count_HVHA, np.sum(pred_int==1)/count_HVLA,
        #                     np.sum(pred_int==2)/count_LVHA, np.sum(pred_int==3)/count_LVLA,
        #                     accuracy_score(Ytest_int, pred_int)] )

        # CM = confusion_matrix( Ytest_int , np.argmax(pred, axis=1) )      
        # _, b = scores(CM )        
        # temp_track.append( b )

    AccTrackDict[_k] = np.array(temp_track)

#%%
ll = ['EEGNet', '+QKV', '+CBAM', '+SE','+M1', '+M2', '+M3']
ll_order = ['Negative', 'Neutral', 'Positive']
fig, ax = plt.subplots(2,4,figsize=(24,12))
count = 0
for _k in ['baseline', 'qkv', 'CBAM', 'SE', 'C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI']:
    _v = AccTrackDict[_k]
    r_num = count//4
    c_num = count%4
    for i in range(3):
        ax[r_num][c_num].plot(interp_grid, _v[:,i], '--', linewidth=4,label = ll_order[i])
    ax[r_num][c_num].plot(interp_grid, _v[:,-2], 'purple', linewidth=4,label = 'Overall')
    
    ax[r_num][c_num].fill_between(interp_grid, _v[:,-2] - _v[:,-1], _v[:,-2] + _v[:,-1], 
                                 color='gray', alpha=0.3, label='std')  
    vv = np.where(_v[:,-1]<0.1, _v[:,-1], 0)
    ax[r_num][c_num].fill_between(interp_grid, _v[:,-2] - vv, _v[:,-2] + vv, 
                                 color='pink', alpha=0.8, label='std(<0.1)')      

    ax[r_num][c_num].set_xlabel(r'$\alpha$', fontdict={'size':20})
    ax[r_num][c_num].set_ylabel('Acc.',fontdict={'size':18})
    ax[r_num][c_num].set_ylim([0,1])
    ax[r_num][c_num].set_title(ll[count],fontdict={'size':24})
    ax[r_num][c_num].grid(axis ='both')
    
    count = count+1
ax[-1][-2].legend(bbox_to_anchor=(1.8, 1.0), fontsize=20)
ax[-1][-1].set_frame_on(False)
ax[-1][-1].set_xticks([])
ax[-1][-1].set_xticklabels([])
ax[-1][-1].set_yticks([])
ax[-1][-1].set_yticklabels([])
plt.subplots_adjust(hspace=0.3)

# %%
