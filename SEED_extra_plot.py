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
                 seg_len=200, lr=1e-3):
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
Extracting channel attention weights for compared models.
They will be used for scalp mappings with matlab
'''
ckpt_path = '/mnt/HDD/Datasets/SEED/ckpt'
subject_selected = 15
model_names = ['baseline','qkv','SE','CBAM','C2A_NNR_DI','C2A_NNR_ID','C2A_NNR_0c']
model_tokens = ['eegnet', 'qkv','SE','CBAM','C2A_NNR_mono_DI','C2A_NNR_mono_ID','C2A_NNR_0c']
# model_names = ['baseline','qkv','SE','CBAM','K_v1']
# model_tokens = ['eegnet', 'qkv','SE','CBAM','kanet_v1']
model_dict = dict(zip(model_names, model_tokens))

#%%
'''
Extracting kernel weights in the depthwise conv layer for visualizing wiht scalp maps
'''

for _s in [15]:
    Collect = []
    for nn_token in model_names:
        W_list = []
        for fld  in range(5):
            model = load_trained(os.path.join(ckpt_path,nn_token), nn_token, subject_selected, count = fld, 
                                num_class=3, seg_len=200, lr=1e-3,model_dict=model_dict)
            # model = load_trained(ckpt_path, nn_token, subject_selected, count = fld, 
            #                      num_class=3, seg_len=200, lr=1e-3,model_dict=model_dict)
            W = model.get_layer('DepthConv').weights
            W_list.append(W[0][0].numpy())
        W_list = np.concatenate(W_list, axis=-1)
        Collect.append(W_list)


    ## Create target array to save, after normalization

    CC = np.array(Collect)[...,0,::2]
    savemat('/mnt/HDD/Datasets/SEED/benchmark_summary/ATT_SEED_7models_S{}.mat'.format(_s), {'CM':CC})

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

with plt.style.context('ggplot'): # compare with resutls from SEED
    fig, ax = plt.subplots(1,3)
    for i in range(3):
        _mean = np.mean(mono_list[i], axis=0)
        _std = np.std(mono_list[i], axis=0)
        ax[i].plot(pts, _mean, 'b-', label='mean')
        ax[i].fill_between(pts, _mean - _std, _mean + _std, color='pink', alpha=0.8, label='std')        
        ax[i].axvline(x=0, color='red', linestyle='--')
        ax[i].set_ylim([0, 1])  
        ax[i].set_title('M{}'.format(i+1))  
        ax[i].legend(loc = 'upper left')  


# %%
'''
Sensitivity on Frequencies
'''

X = loadmat( os.path.join(data_path, 'S{:02d}_E01.mat'.format(subject_selected)) )['segs'].transpose([2,1,0])
chns = np.arange(62)
# X = zscore(X[...,chns], axis=1)
chns_token = '{:02d}'.format(len(chns))
Y = loadmat( os.path.join(data_path, 'Label.mat') )['seg_labels'][0]+1


#%%
from Utils import batch_band_pass
data_seq = [ batch_band_pass(X, 0.1, hp, 200) for hp in [10*i for i in range(1,10)]]
data_seq.append(X)
data_seq = [zscore(_d, axis=1) for _d in data_seq]

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


plt.figure()
freq_grid = [10*i for i in range(1,11)]
# ll = ['EEGNet', '+QKV',  '+SE', '+CBAM', '+M1', '+M2', '+M3']
ll = ['EEGNet', '+QKV',   '+CBAM', '+SE','+M1', '+M2', '+M3']
count = 0
# for _k, _v in freq_dict.items():
#     plt.plot(freq_grid, _v[:,0], 'd--',label = ll[count])
#     count += 1
# plt.ylim([0.25, 1.0])
# plt.xlabel('Hz')
# plt.ylabel('Acc')
# plt.legend(loc = 'upper left')

# for _k in ['baseline', 'qkv', 'SE', 'CBAM', 'C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI']:
for _k in ['baseline', 'qkv', 'CBAM', 'SE', 'C2A_NNR_0c', 'C2A_NNR_ID', 'C2A_NNR_DI']:
    plt.plot(freq_grid, freq_dict[_k][:,0], 'd--',label = ll[count])
    count += 1
plt.ylim([0.25, 1.0])
plt.xlabel('Hz')
plt.ylabel('Acc')
plt.legend(loc = 'upper left')
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
