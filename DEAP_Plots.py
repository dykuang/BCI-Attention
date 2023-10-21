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
from matplotlib import patches
from matplotlib.patheffects import withStroke
import seaborn as sns
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

#%%
'''
Load Detailed scores
'''

model_names = ['EEGNet','SE', 'CBAM', 'Mnt_DI', 'Mnt_ID', 'Mnt_no', 'QKV','KAM']
# model_names = ['EEGNet','SEER', 'DCN', 'DCN_KAM']
# model_names = ['EEGNet', 'QKV', 'SE', 'CBAM', 'KAM']
summary_path = 'mnt/HDD/Benchmarks/DEAP/summary'
exp_type =2
val_mode = 'random'
summary_dict = dict.fromkeys(model_names)
subjects = [ i for i in range(1,33) if i!=23 ]

for _m in model_names:
    temp_dict = dict.fromkeys(subjects)
    for _s in subjects:
        S_temp = np.load('/mnt/HDD/Benchmarks/DEAP/summary/S{:02d}_{}_type{}_{}.npy'.format(_s, _m, exp_type, val_mode))
        SW_temp = np.load('/mnt/HDD/Benchmarks/DEAP/summary/SW{:02d}_{}_type{}_{}.npy'.format(_s, _m, exp_type, val_mode))
        CM_temp = np.load('/mnt/HDD/Benchmarks/DEAP/summary/CM_S{:02d}_{}_type{}_{}.npy'.format(_s, _m, exp_type, val_mode))
        temp_dict[_s] = {'Score': S_temp,
                         'Weighted_Score': SW_temp,
                         'CM': CM_temp
                         }
    
    summary_dict[_m] = temp_dict

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

#%%
'''
Perform some statistical test
'''
from scipy.stats import ttest_ind, ttest_rel, levene
P_EEG  = get_perform_overall('EEGNet')
P_CBAM = get_perform_overall('CBAM')
P_SE = get_perform_overall('SE')
P_M0 = get_perform_overall('Mnt_no')
P_M_ID = get_perform_overall('Mnt_ID')
P_M_DI = get_perform_overall('Mnt_DI')
P_QKV  = get_perform_overall('QKV')
P_KAM  = get_perform_overall('KAM')

ttest_rel(P_EEG[0][:,-1], P_M_DI[0][:,-1], alternative='less') #>EEGNet
ttest_rel(P_M_DI[0][:,-1], P_SE[0][:,-1], alternative='less') #~ SE
ttest_rel(P_M0[0][:,-1], P_M_DI[0][:,-1], alternative='less') # ~ no constraint
ttest_rel(P_M_ID[0][:,-1], P_M_DI[0][:,-1], alternative='less') # >ID
ttest_rel(P_CBAM[0][:,-1], P_M_DI[0][:,-1], alternative='less') # > CBAM

#%%
F1 = []
F1_std = []
model_compared = ['EEGNet', 'QKV', 'SE', 'CBAM', 'KAM']
for _m in model_compared:
    temp_m = get_perform_overall(_m)[0][:,-1]
    temp_std= get_perform_overall(_m)[1][:,-1]
    F1.append(temp_m)
    F1_std.append(temp_std)

F1 = np.array(F1)
F1_std = np.array(F1_std)
best_model_per_sub = np.argmax(F1, axis=0)

palette = sns.color_palette("tab10", 10).as_hex()

best_F1_per_sub = [F1[best_model_per_sub[i], i]*100 for i in range(31)]
best_F1_std_per_sub = [F1_std[best_model_per_sub[i], i]*100 for i in range(31)]

ylabel = ['S{}'.format(i) for i in range(1,33) if i != 23]
palette = sns.color_palette("tab10", 10).as_hex()
color = [palette[i] for i in best_model_per_sub]

#%%

def make_hbar(ylabel, counts, err, color, x_lb=90): # be aware of passing reverse order
    # The positions for the bars
    # This allows us to determine exactly where each bar is located
    y = [i * 0.9 for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(y, counts, height=0.55, align="edge", color=color)
    ax.errorbar(counts, [_y+0.3 for _y in y], xerr = err, fmt ='o', color='k')
    # ax.xaxis.set_ticks([i * 5 for i in range(0, 12)])
    # ax.xaxis.set_ticklabels([i * 5 for i in range(0, 12)], size=16, fontfamily="Econ Sans Cnd", fontweight=100)

    ax.yaxis.set_ticks([_y+0.3 for _y in y])
    ax.yaxis.set_ticklabels(ylabel, size=16, fontfamily="Econ Sans Cnd", 
                            fontweight=100)
    ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)

    ax.set_xlim((x_lb, 100))
    ax.set_ylim((0, len(counts) * 0.9 - 0.2))

    # Set whether axis ticks and gridlines are above or below most artists.
    ax.set_axisbelow(True)
    ax.grid(axis = "x", color="#A8BAC4", lw=1.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_lw(1.5)
    # This capstyle determines the lines don't go beyond the limit we specified
    # see: https://matplotlib.org/stable/api/_enums_api.html?highlight=capstyle#matplotlib._enums.CapStyle
    ax.spines["left"].set_capstyle("butt")

    # Hide y labels
    # ax.yaxis.set_visible(False)
    PAD = 0.3
    for count, _e, y_pos in zip(counts, err, y):
        x = x_lb
        color = "white"
        path_effects = None
        # if count < 8:
        #     x = count
        #     color = BLUE    
        #     path_effects=[withStroke(linewidth=6, foreground="white")]
        
        ax.text(
            x + PAD, y_pos + 0.5 / 2, '{:.02f}$\pm${:.02f}%'.format(count,_e), 
            color=color, fontfamily="DejaVu Sans", fontsize=18, va="center",
            path_effects=path_effects
        ) 

    # legend_elements = [patches.Patch(facecolor=c, edgecolor=None, label='tt{}'.format(i)) for i,c in enumerate(color)]

    # ax.legend(handles=legend_elements, loc='right')

make_hbar(ylabel[:8][::-1], best_F1_per_sub [:8][::-1], best_F1_std_per_sub[:8][::-1], color[:8][::-1],x_lb=85)
make_hbar(ylabel[8:16][::-1], best_F1_per_sub [8:16][::-1], best_F1_std_per_sub[8:16][::-1], color[8:16][::-1])
make_hbar(ylabel[16:24][::-1], best_F1_per_sub [16:24][::-1], best_F1_std_per_sub[16:24][::-1], color[16:24][::-1])
make_hbar(ylabel[24:][::-1], best_F1_per_sub [24:][::-1], best_F1_std_per_sub[24:][::-1], color[24:][::-1])    


#%%
'''
Plots about trained models
'''
from Models import *

def load_trained(ckpt_path, nn_token, subject, exp_type, val_mode='random', 
                 count = 0, num_class=3, seg_len=128, lr=1e-3):
    if nn_token == 'EEGNet':
        model = EEGNet(nb_classes = num_class, Chans = 32, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    learning_rate = lr)
    elif nn_token == 'SE':
        model = CANet(nb_classes= num_class, Chans = 32, Samples = seg_len, attention_module = 'se_block',
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token =='CBAM':
        model = CANet(nb_classes = num_class, Chans = 32, Samples = seg_len, attention_module='cbam_block',
                    dropoutRate = 0.5, kernLength = 5, F1 = 8,
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token == 'Mnt_ID':
        model = MTNet(nb_classes= num_class, Chans = 32, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    penalty_rate= 0.1, mono_mode ='ID',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token == 'Mnt_DI':
        model = MTNet(nb_classes= num_class, Chans = 32, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    penalty_rate= 0.1, mono_mode ='DI',
                    optimizer = Adam, learning_rate = lr)
    elif nn_token == 'Mnt_no':
        model = MTNet(nb_classes = num_class, Chans = 32, Samples = seg_len,
                    dropoutRate = 0.5, kernLength = 5, F1 = 8,
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    penalty_rate = 0.0, mono_mode = None,
                    optimizer = Adam, learning_rate = lr)
    elif nn_token == 'QKV':
        model = QKVNet(nb_classes = num_class, Chans = 32, Samples = seg_len, 
                  dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                  D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                  optimizer = Adam, learning_rate = lr)     
    elif nn_token == 'KAM':
        model = KANet(nb_classes = num_class, Chans = 32, Samples = seg_len, 
                    dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                    optimizer = Adam, learning_rate = lr)   
    elif nn_token in ['DCN']:
        model = DeepConvNet(nb_classes = num_class, Chans = 32, Samples = seg_len,
                    dropoutRate = 0.25, attention_type = None,
                    optimizer = Adam, learning_rate = lr)
    elif nn_token in ['DCN_KAM']:
        model = DeepConvNet(nb_classes = num_class, Chans = 32, Samples = seg_len,
                    dropoutRate = 0.25, attention_type = 'KAM',
                    optimizer = Adam, learning_rate = lr)
    else:
        assert 'nn_token not recognized.'

    model.load_weights(os.path.join(ckpt_path, 'S{:02d}_ckpt_{}_type{}_count{}'.format(subject, nn_token, exp_type, count)))

    return model

#
#%%
'''
Extracting channel attention weights for compared models:
'''
ckpt_path = '/mnt/HDD/Benchmarks/DEAP/ckpt'
subject_selected = 32
model_compared = ['EEGNet', 'QKV',  'CBAM', 'SE', 'Mnt_no', 'Mnt_ID', 'Mnt_DI']
# model_compared = ['EEGNet', 'QKV', 'SE', 'CBAM', 'KAM']
# model_compared = ['EEGNet', 'QKV',  'CBAM', 'SE', 'Mnt_no', 'Mnt_ID', 'Mnt_DI','KAM']

#%%

# for _s in [7, 8, 12, 24, 31]:

#     Collect = []

#     for nn_token in model_compared:
#         W_list = []
#         for fld  in range(10):
#             model = load_trained(ckpt_path, nn_token, _s, exp_type, val_mode, 
#                                 count = fld, num_class=4, seg_len=128, lr=1e-3)
#             W = model.get_layer('DepthConv').weights
#             W_list.append(W[0][0].numpy()) #(32, 8, 2)
#         W_list = np.concatenate(W_list, axis=-1) #(32,  8, 20)
#         Collect.append(W_list)


#     ## Create target array to save, after normalization

#     CC = np.array(Collect)[...,0,::2]
#     # savemat('/mnt/HDD/Benchmarks/DEAP/ATT_DEAP_7models_S{:02d}.mat'.format(_s), {'CM':CC})
#     savemat('/mnt/HDD/Benchmarks/DEAP/ATT_DEAP_5models_S{:02d}.mat'.format(_s), {'CM':CC})
#     # savemat('/mnt/HDD/Benchmarks/DEAP/ATT_DEAP_QKV_S{:02d}.mat'.format(_s), {'CM':CC})


for _s in [8, 12, 16, 24, 32]:

    Collect = []

    for nn_token in model_compared:
        W_list = []
        for fld  in range(10):
            model = load_trained(ckpt_path, nn_token, _s, exp_type, val_mode, 
                                count = fld, num_class=4, seg_len=128, lr=1e-3)
            W = model.get_layer('DepthConv').weights
            W_list.append(W[0][0].numpy()) #(32, 8, 2)
        W_list = np.concatenate(W_list, axis=-1) #(32,  8, 20)
        Collect.append(W_list.reshape(32, -1))


    ## Create target array to save, after normalization

    CC = np.array(Collect)
    # savemat('/mnt/HDD/Benchmarks/DEAP/ATT_DEAP_7models_S{:02d}.mat'.format(_s), {'CM':CC})
    savemat('/mnt/HDD/Benchmarks/DEAP/DEAP_scalp_all_models_S{:02d}.mat'.format(_s), {'CM':CC})


#%%
'''
Try exploring the weight correlation among sensor locations
'''
from scipy.io import loadmat

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
import pickle
with open('./ch_pos_1020.pkl', 'rb') as pkl:
    pos_dict = pickle.load(pkl)

if subject_selected<23:
    ch_list = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5',
                'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
                'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
                'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
                'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
                'Fz', 'Cz'
    ]; 
else:
    ch_list = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1',
                'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
              'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4',
              'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
              'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',
              'PO4', 'O2'
              ]

XY = []
for ch in ch_list:
    if ch in pos_dict.keys():
        XY.append(pos_dict[ch][:2])
XY = np.array(XY)
              
scalp_weights = loadmat('/mnt/HDD/Benchmarks/DEAP/DEAP_scalp_all_models_S{:02d}.mat'.format(subject_selected))['CM']

# For each model, get the most correlated channel and max correlated channel pair

# cor = np.corrcoef(scalp_weights[0])

for _w in scalp_weights:
    cor = np.corrcoef(_w)
    link_cor_2d(cor, XY)
# plt.figure()
# XY = []
# for _k, _v in pos_dict.items():
#     plt.text(_v[0], _v[1], _k)
# plt.xlim([-0.2,0.2])
# plt.ylim([-0.2, 0.2])

# %%
'''
umap embedding of feature space
'''
import umap
import seaborn as sns
from tensorflow.keras.utils import to_categorical

def make_segs(data, seg_len, stride):
    t_len = data.shape[1]
    segs = np.stack([data[:,i*stride:i*stride+seg_len,:] for i in range(t_len//stride) if i*stride+seg_len<=t_len], axis= 1)
    # print(segs.shape)
    return segs.reshape((-1, seg_len, data.shape[-1]))

def get_dense_output(model, X):
    try:
        f_model = Model(model.input, model.get_layer('last_dense').output)
    except:
        f_model = Model(model.input, model.get_layer('dense').output)
    return f_model.predict(X)

def get_best_model(ckpt_path, nn_token, subject_selected, exp_type=2, val_mode = 'random'):
    score = summary_dict[nn_token][subject_selected]['Weighted_Score']
    fld = np.argmax(score[:,-1])
    model = load_trained(ckpt_path, nn_token, subject_selected, exp_type, val_mode, 
                            count = fld, num_class=4, seg_len=128, lr=1e-3)
    return model    

#%%
path = '/mnt/HDD/Datasets/DEAP/s{:02d}.mat'.format( int(subject_selected) )
raw = loadmat(path)

data = raw['data'][:,:32,:] # only using eeg recordings
label = raw['labels'] # valence, arousal, dominance, liking

# data_N = zscore(data, axis= -1)
# data_N = data_N.transpose((0, 2, 1))
data = data.transpose((0, 2, 1))
data_N = data/np.max(abs(data[:,:128*3,:]), axis=1, keepdims=True)

label_V_TF = label[:,0]>5
label_V = to_categorical(label_V_TF)
label_A_TF = label[:,1]>5
label_A = to_categorical(label_A_TF)
label_VA_TF = -1* np.ones(len(label_A))
label_VA_TF[label_V_TF & label_A_TF] = 0  #HVHA
label_VA_TF[np.logical_and(label_V_TF, np.logical_not(label_A_TF))] = 1 #HVLA
label_VA_TF[np.logical_and(np.logical_not(label_V_TF), label_A_TF)] = 2 #LVHA
label_VA_TF[np.logical_not(label_V_TF) & np.logical_not(label_A_TF)] = 3 #LVLA
label_VA = to_categorical(label_VA_TF)

label_meaning = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
segs_to_check = make_segs(data_N, 128, 128)


#%%
base_model = get_best_model(ckpt_path, 'EEGNet', subject_selected, exp_type=2, val_mode = 'random')
# base_model = get_best_model(ckpt_path, 'KAM', subject_selected, exp_type=2, val_mode = 'random')
baseline_data = get_dense_output(base_model, segs_to_check )
baseline_label =  np.repeat(label_VA_TF, len(segs_to_check)//40, axis=0)
#%%
mapper = umap.UMAP(n_neighbors=100, n_components=2, metric='euclidean', 
                   spread=1.0, min_dist=0.2, local_connectivity=1.0,
                   output_metric='euclidean', init='spectral', 
                   densmap=False, random_state = 12345)

embeded=[]
embeded.append( mapper.fit_transform(baseline_data) )
plt.figure()
sns.scatterplot(x = embeded[0][:,0], y = embeded[0][:,1], hue = baseline_label)



#%%
'''
altering alpha value in a trained module and check its umap embedding
'''
umap_model = get_best_model(ckpt_path, 'KAM', subject_selected, exp_type=2, val_mode = 'random')
umap_input_data = get_dense_output(umap_model, segs_to_check )
embeded_track = []
mapper = umap.UMAP(n_neighbors=100, n_components=2, metric='euclidean', 
                   spread=1.0, min_dist=0.2, local_connectivity=1.0,
                   output_metric='euclidean', init='spectral', 
                   densmap=False, random_state = 12345)

mapper.fit(umap_input_data)

alpha_to_check = np.linspace(0, 0.1, 11)
for run_alpha in alpha_to_check:
    umap_model.get_layer('Katt').set_weights(np.array([[run_alpha]]))
    embeded_track.append(get_dense_output(umap_model, segs_to_check))

#%%
'''
plotly animated 2d
'''
import plotly.express as px
# df = px.data.gapminder()
# fig= px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
#                 size="pop", color="continent", hover_name="country",
#                 log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

# make a dataframe for convenience
import pandas as pd
coord_data = np.concatenate([mapper.transform(t) for t in embeded_track],axis=0)
embeded_df = pd.DataFrame(data=coord_data, columns=['UMAP-x', 'UMAP-y'])
embeded_df['alpha'] = np.repeat(alpha_to_check, len(embeded_track[0]))
label_name = [label_meaning[int(j)] for j in baseline_label]
embeded_df['label'] = np.tile(label_name, len(alpha_to_check))

fig= px.scatter(embeded_df, x="UMAP-x", y="UMAP-y", animation_frame="alpha", 
                # animation_group="label",
                # size="pop", 
                color="label", 
                # hover_name="label",
                log_x=False, size_max=55, 
                # range_x=[100,100000], range_y=[25,90]
                )

fig.show(renderer='browser')


#%%
cpr_model = [base_model]
for _m in model_compared[1:]:
    cpr_model.append( get_best_model(ckpt_path, _m, subject_selected, exp_type=2, val_mode = 'random') )
    # cpr_data = get_dense_output(cpr_model[-1], segs_to_check )  
    # embeded.append( mapper.transform(cpr_data) )

# fig = plt.figure(figsize=(12,8))
# for i in range(1, 7):
#     # _r = i//3 + 1
#     # _c = i%3 + 1
#     axes = fig.add_subplot(2,3,i)
#     sns.scatterplot(x = embeded[i-1][:,0], y = embeded[i-1][:,1], hue = baseline_label)


#%%
'''
plot the monotocity
'''
pts = np.linspace(-1.0, 1.0, 21)
def get_att_map(model):

    x_In = layers.Input((1,1,1), name='map_in')
    x = model.get_layer('att_mono').D1(x_In)
    x= model.get_layer('att_mono').D2(x)
    x= model.get_layer('att_mono').BN(x)
    x_Out= model.get_layer('att_mono').Pred(x)

    return Model(x_In, x_Out)

#%%
fig, ax = plt.subplots(1,3)

att = get_att_map(cpr_model[-3])
cc = att.predict(pts.reshape([-1,1,1,1]))[:,0,0,0]
ax[0].plot(pts, cc)
ax[0].axvline(x=0, color='red', linestyle='--')
ax[0].set_ylim([0, 1])

att = get_att_map(cpr_model[-2])
cc = att.predict(pts.reshape([-1,1,1,1]))[:,0,0,0]
ax[1].plot(pts, cc)
ax[1].axvline(x=0, color='red', linestyle='--')
ax[1].set_ylim([0, 1])

att = get_att_map(cpr_model[-1])
cc = att.predict(pts.reshape([-1,1,1,1]))[:,0,0,0]
ax[2].plot(pts, cc)
ax[2].axvline(x=0, color='red', linestyle='--')
ax[2].set_ylim([0, 1])

#%%
# check for the mean and std of learned monotoncity
mono_list = []
for nn_token in ['Mnt_no', 'Mnt_ID', 'Mnt_DI']:
    temp_list = []
    for fld  in range(10):
        model = load_trained(ckpt_path, nn_token, subject_selected, exp_type, val_mode, 
                             count = fld, num_class=4, seg_len=128, lr=1e-3)
        att = get_att_map(model)
        cc = att.predict(pts.reshape([-1,1,1,1]))[:,0,0,0]
        temp_list.append(cc)
    mono_list.append(np.array(temp_list))

with plt.style.context('ggplot'): # compare with resutls from SEED
    fig, ax = plt.subplots(1,3, figsize=(6,4))
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
'''make some sample feature plots'''

feature_list = []
for nn_token in ['Mnt_no', 'Mnt_ID', 'Mnt_DI']:
    temp = []
    for fld  in range(10):
        model = load_trained(ckpt_path, nn_token, subject_selected, exp_type, val_mode, 
                             count = fld, num_class=4, seg_len=128, lr=1e-3)
        fmap = Model(model.input,  
                     model.get_layer('att_mono').output[0] 
                     - model.get_layer('SepConv-1').output[...,0,:] ) #get the difference
        temp.append(fmap.predict(segs_to_check))

    feature_list.append(np.array(temp))

feature_list = np.array(feature_list) # (monotype, folds, samples, ...)
feature_list_m = np.mean(feature_list, axis= 1) # mean over folds

rnd_ind = 10

Feature2Plot = feature_list_m[:,[np.where(baseline_label ==i)[0][rnd_ind] for i in range(4)],...] #(monotype, label, ...)
# Feature2Plot = Feature2Plot/np.max(np.abs(Feature2Plot), keepdims=True) # normalization 

#%%
with plt.style.context('ggplot'): 
    fig, ax = plt.subplots(4,3,figsize=(8,6))
    for i in range(4):
        for j in range(3):
            _im = ax[i][j].imshow(Feature2Plot[j,i].T, cmap = 'jet')
            ax[i][j].set_axis_off()
    plt.colorbar(_im, ax=ax.ravel().tolist() ) 
            



# %%
'''
Gradcam
'''
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([0, 1, 2, 3])

# Instead of using CategoricalScore object,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    return (output[0][0], output[1][1], output[2][2], output[3][3])

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

# # prepare samples
# seg_pred_score = cpr_model[0].predict(segs_to_check)
# seg_pred_label = np.argmax(seg_pred_score, axis=-1)

# idx2vis_correct = []
# idx2vis_wrong = []
# for i in range(4):
#     idx2vis_correct.append( np.where( np.logical_and( seg_pred_label == baseline_label, seg_pred_label==i))[0] )
#     idx2vis_wrong.append( np.where( np.logical_and( seg_pred_label != baseline_label, seg_pred_label==i))[0]  )
idx2vis_correct, idx2vis_wrong = get_CM_idx(cpr_model[0], segs_to_check, baseline_label)

#%%
cam_selected_idx = np.array([idx2vis_correct[i][0]  for i in range(4)])
X_cam = np.array(segs_to_check[cam_selected_idx])


def make_GradAMPP(model, X, score_func=score_function, penultimate_layer=-2):

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model,
                            model_modifier = None,
                            #   model_modifier=replace2linear,  #if dense layer has other activations than linear
                            clone=True)

    # Generate heatmap with GradCAM++
    cam = gradcam(score,
                  X,
                  penultimate_layer=penultimate_layer)

    return cam

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

cam = make_GradAMPP(cpr_model[0], X_cam[...,None])

# Render

#%%
'''
Corruption by low-pass filtering, noises or other kinds of attacks

pick a correctly predicted sample, corrupt it with different levels of the same attack, 
trace the prediction.

alternatives: band passing with different bands is also possible
'''
from Utils import batch_band_pass

segs_to_check_tst = make_segs(data_N[:,6000:,:], 128, 128)
Ytest = np.repeat(label_VA, segs_to_check_tst.shape[0]//40, axis=0)
# data_seq = [ batch_band_pass(segs_to_check_tst, 0.1, hp, 128) for hp in [60, 50, 40, 30, 20, 10] ]
# data_seq.insert(0, segs_to_check_tst)

rej_band = [(l, l+8) for l in range(8,56,8)]
rej_band.append((56, 63.99))
rej_band.insert(0, (1, 8))
data_seq = [ batch_band_pass(segs_to_check_tst, lp, hp, 128, btype='bandstop') for 
             (lp,hp) in rej_band ]
data_seq.append(segs_to_check_tst)

#%%
freq_dict = {}

for _name, _m in zip(model_compared, cpr_model):
    freq_p = []
    for _data in data_seq:
        pred = _m.predict(_data)
        # CM = confusion_matrix( baseline_label, np.argmax(pred, axis=1))
        CM = confusion_matrix( np.argmax(Ytest,axis=1), np.argmax(pred, axis=1) )
        # print(CM)
        
        _, b = scores(CM )
        freq_p.append(b )
    freq_dict[_name] = np.array(freq_p)   
 
# plt.figure()
# freq_grid = [10, 20, 30, 40, 50, 60, 64]
# ll = ['EEGNet', '+QKV',   '+CBAM', '+SE','+M1', '+M2', '+M3']
# count = 0
# for _k, _v in freq_dict.items():
#     if _k != 'KAM':
#         plt.plot(freq_grid, _v[::-1,0], 'd--',label = ll[count])
#     count += 1
# plt.ylim([0.25, 1.0])
# plt.xlabel('Hz')
# plt.ylabel('Acc')
# plt.legend(loc = 'upper left')

# plt.figure()
# freq_grid = [10, 20, 30, 40, 50, 60, 64]
# ll = ['EEGNet', '+QKV',  '+SE', '+CBAM', '+KAM']
# count = 0
# for _k, _v in freq_dict.items():
#     plt.plot(freq_grid, _v[::-1,0], 'd--',label = ll[count])
#     count += 1
# plt.ylim([0.25, 1.0])
# plt.xlabel('Hz')
# plt.ylabel('Acc')
# plt.legend(loc = 'upper left')

##================================
# For bandreject case
##=================================
# with plt.style.context('ggplot'):    
plt.figure()
# freq_grid = [10, 20, 30, 40, 50, 60, 64]
# ll = ['EEGNet', '+QKV',  '+SE', '+CBAM', '+KAM']
ll = ['EEGNet', '+QKV', '+CBAM', '+SE','+M1', '+M2', '+M3']
count = 0
for _k, _v in freq_dict.items():
    plt.plot(_v[:,0], 'd--',label = ll[count])
    count += 1
plt.ylim([0.25, 1.0])
plt.xticks(np.arange(_v.shape[0]), 
        labels=['1-8', '8-16', '16-24', '24-32', '32-40', '40-48',
                '48-56', '56-63.99', 'original'], rotation=-45)
plt.legend(loc = 'lower right')
plt.xlabel('Hz')
plt.ylabel('Acc')
plt.grid()

#%%
# attack by linear interpolating between two curves

Correct_list = []
# for i in [0, 1, 2, 5]:
for i in range(len(cpr_model)):
    temp_correct, _ = get_CM_idx(cpr_model[i], segs_to_check, baseline_label)
    Correct_list.append(temp_correct)

def common(lst1, lst2): 
    return list(set(lst1) & set(lst2))

Common_idx = []
for j in range(len(label_meaning)): #j for labels
    a = Correct_list[0][j].copy()
    for i in range(1,len(cpr_model)): #i for models
        a = common(a, Correct_list[i][j])
    Common_idx.append(a)

for i in range(len(label_meaning)):
    print(len(Common_idx[i]))

def morphed_curve(cA, cB, grid=[0, 0.25, 0.5, 0.75, 1.0]):
    track = []
    for _h in grid:
        track.append((1-_h)*cA + _h*cB)
    return track


segs_for_morph = np.array(segs_to_check[[c[0] for c in Common_idx]])

#%%
from itertools import combinations

interp_grid = np.linspace(0,1,101)
# trackDict = dict.fromkeys(['EEGNet', 'QKV', 'CBAM', 'SE', 'Mnt_DI', 'Mnt_no', 'Mnt_ID'])
# crossDict = dict.fromkeys(['EEGNet', 'QKV', 'CBAM', 'SE', 'Mnt_DI', 'Mnt_no', 'Mnt_ID'])

trackDict = dict.fromkeys(['EEGNet', 'QKV', 'CBAM', 'SE', 'KAM'])
crossDict = dict.fromkeys(['EEGNet', 'QKV', 'CBAM', 'SE', 'KAM'])
for _k in trackDict.keys():
    temp_track = []
    temp_cross = []
    i = model_compared.index(_k)
    for _p in combinations(np.arange(len(segs_for_morph)), 2):      
        track = morphed_curve(segs_for_morph[_p[0]], segs_for_morph[_p[1]], interp_grid)
        temp_track.append( cpr_model[i].predict(np.array(track)[...,None]) )
        temp_cross.append( (interp_grid[1]-interp_grid[0])*np.argmin( abs(temp_track[-1][:,_p[0]] - temp_track[-1][:,_p[1]]) )  )
    trackDict[model_compared[i]] = temp_track
    crossDict[model_compared[i]] = temp_cross

# for i in trackDict['EEGNet']:
#     plt.figure()
#     plt.plot(interp_grid, i, '+--')


#%%
def trig_abl_plot(cross_track, make_plot= False):
    '''
    For N=4, needs modifications for other values of N
    '''
    theta = np.array([90, 210, 330, 90])
    theta = theta*np.pi/180.0
    Data2Plot = [cross_track[:3], 
                 [1-cross_track[0]]+cross_track[3:5], 
                 [1-cross_track[1], 1-cross_track[3], cross_track[5]],
                 [1-cross_track[2], 1-cross_track[4], 1-cross_track[5]]
                 ]
    labels = [ ['HVHA', 'HVLA', 'LVHA', 'LVLA'],
               ['HVLA', 'HVHA', 'LVHA', 'LVLA'],
               ['LVHA', 'HVHA', 'HVLA', 'LVLA'],
               ['LVLA', 'HVHA', 'HVLA', 'LVHA']  
            ]
    
    if make_plot:
        colors = ['r', 'g', 'cyan', 'y']
        plt.figure()
        for i in range(4):
            ax = plt.subplot(1,4,i+1, projection='polar')
            temp = np.append(Data2Plot[i], Data2Plot[i][0]) #make a closed loop
            ax.plot(theta, temp,colors[i])
            ax.fill(theta, temp,colors[i], alpha=0.3)
            ax.set_xticks(theta[:3])
            ax.set_xticklabels(labels[i][1:], y=0.1)
            ax.set_ylim([0, 1.0])
            ax.set_title('Origin-{} \n {:.02f}'.format(labels[i][0], np.sum(temp[:3])), color=colors[i], size=10)
    
    return Data2Plot

# for _k, _v in crossDict.values():
#     trig_abl_plot(_v)
#     print('{}:{}'.format(_k, np.sum(_v)))

_ = trig_abl_plot(crossDict['EEGNet'], make_plot=True)

#%%
'''
Donut plot for prediction robust region.
'''
# score = [[1.32, 1.16, 1.70, 1.82],
#         [1.67, 1.11, 1.39, 1.83],
#         [1.81,1.05,1.44,1.70],
#         [1.94,1.24,1.32,1.50],
#         [1.63,1.22,1.56,1.59],
#         [1.56, 1.14, 1.71, 1.59]] 

score= []
for _m in model_compared:
    temp = trig_abl_plot(crossDict[_m], make_plot=False)
    score.append(np.sum(temp, axis=-1))

def make_donut(sizes, nn_token):
        labels = ['HVHA', 'HVLA','LVHA','LVLA']
        
        # colors
        colors = ['r', 'g', 'cyan', 'yellow']
        
        # explosion
        explode = (0.05, 0.05, 0.05, 0.05)

        plt.figure()       
        # Pie Chart
        plt.pie(sizes, colors=colors, labels=labels,
                autopct='%1.1f%%', pctdistance=0.7,
                explode=explode, textprops= {'fontsize': 12})
        
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.40, fc='white')
        fig = plt.gcf()
        
        # Adding Circle in Pie chart
        fig.gca().add_artist(centre_circle)

        # Adding center text
        plt.text(-0.25,-0.05,nn_token,fontsize=16)

# make_donut(score[1], '+CBAM')

#%%
def pred_trace_plot(pred_track, attach_legend=False, title=None):
    theta = np.array([0, 90, 180, 270, 0])
    theta = theta*np.pi/180.0
    colors = ['r', 'g', 'b', 'y', 'orange', 'purple']
    labels = ['HVHA', 'HVLA', 'LVHA', 'LVLA']

    legend = ['HVHA--HVLA', 'HVHA--LVHA', 'HVHA--LVLA',
              'HVLA--LVHA', 'HVLA--LVLA', 'LVHA--LVLA']

    plt.figure()
    ax = plt.subplot(1,1,1, projection='polar')
    for i in range(6):
        ll = np.argmax(pred_track[i], axis=1)
        trace = []
        phi = []
        for count, j in enumerate(ll):
            trace.append(pred_track[i][count,j])
            phi.append(theta[j])

        ax.plot(phi, trace, colors[i], label=legend[i] , linewidth = 4)
    ax.plot(theta, [1,1,1,1,1], 'k-')
    ax.set_xticks(theta[:4])
    ax.set_xticklabels(labels, y=0.0)
    ax.set_ylim([0, 1.0])
    if attach_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_title(title, loc='left', size=16)

pred_trace_plot(trackDict['EEGNet'], title='EEGNet')
# pred_trace_plot(trackDict['CBAM'], title = '+CBAM')
# pred_trace_plot(trackDict['SE'], title = '+SE')
# pred_trace_plot(trackDict['Mnt_DI'], title = '+KAM')



#%%
'''
Patial dependences via Jacobian
'''
def get_partial_on_depthconv(model, label_idx, X_samples):
    with tf.GradientTape() as tape:
        W = model.get_layer('DepthConv').weights
        tape.watch(W)
        y = model(X_samples)[:, label_idx]
        jac = tape.jacobian(y, W)

    return jac

J = get_partial_on_depthconv(cpr_model[0], 0, segs_to_check[idx2vis_correct[0]])
# J is of shape (..., 1, 32(kernel length), 8(kernel num), 2(depth_mulitplier) ) 


# %%
'''
track the accuracy change while morphing from zeros to X linearly on the amplitude
'''
from sklearn.metrics import accuracy_score
interp_grid = np.linspace(0,1,21)
AccTrackDict = dict.fromkeys(['EEGNet', 'QKV', 'CBAM', 'SE', 'Mnt_DI', 'Mnt_no', 'Mnt_ID'])
Ytest_int = np.argmax(Ytest,axis=1)
count_HVHA = np.sum(Ytest_int==0)  #HVHA
count_HVLA = np.sum(Ytest_int==1)  #HVLA
count_LVHA = np.sum(Ytest_int==2)  #LVHA
count_LVLA = np.sum(Ytest_int==3)  #LVLA
count_all = Ytest_int.size

for _k in AccTrackDict.keys():
    temp_track = []
    i = model_compared.index(_k)
    for alpha in interp_grid:
        pred = cpr_model[i].predict(alpha*segs_to_check_tst)
        pred_int = np.argmax(pred,axis=1)
        CM = confusion_matrix( Ytest_int , pred_int )

        temp_track.append( [CM[0,0]/count_HVHA , CM[1,1]/count_HVLA ,
                            CM[2,2]/count_LVHA , CM[3,3]/count_LVLA ,
                            accuracy_score(Ytest_int, pred_int)
                            ] )
        temp_track[-1] += [(4/3)**0.5*(count_HVHA/count_all*(temp_track[-1][0]-temp_track[-1][-1])**2
                          + count_HVLA/count_all*(temp_track[-1][1]-temp_track[-1][-1])**2
                          + count_LVHA/count_all*(temp_track[-1][2]-temp_track[-1][-1])**2
                          + count_LVLA/count_all*(temp_track[-1][3]-temp_track[-1][-1])**2)**0.5]#append the weighted std

        # temp_track.append( [np.sum(pred_int==0)/count_HVHA, np.sum(pred_int==1)/count_HVLA,
        #                     np.sum(pred_int==2)/count_LVHA, np.sum(pred_int==3)/count_LVLA,
        #                     accuracy_score(Ytest_int, pred_int)] )

        # CM = confusion_matrix( Ytest_int , np.argmax(pred, axis=1) )      
        # _, b = scores(CM )        
        # temp_track.append( b )

    AccTrackDict[model_compared[i]] = np.array(temp_track)

#%%
ll = ['EEGNet', '+QKV', '+CBAM', '+SE','+M1', '+M2', '+M3']
ll_order = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
fig, ax = plt.subplots(2,4,figsize=(24,12))
count = 0
for _k, _v in AccTrackDict.items():
    r_num = count//4
    c_num = count%4
    for i in range(4):
        ax[r_num][c_num].plot(interp_grid, _v[:,i], '--', linewidth=4,label = ll_order[i])
    ax[r_num][c_num].plot(interp_grid, _v[:,-2], linewidth=4, label = 'Overall')
    
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
ax[-1][-2].legend(bbox_to_anchor=(1.8, 1.0),fontsize=20)
ax[-1][-1].set_frame_on(False)
ax[-1][-1].set_xticks([])
ax[-1][-1].set_xticklabels([])
ax[-1][-1].set_yticks([])
ax[-1][-1].set_yticklabels([])
plt.subplots_adjust(hspace=0.3)
# plt.ylim([0.0, 1.0])
# plt.legend(loc = 'lower right')
# plt.xlabel('$\alpha$')
# plt.ylabel('Acc.')
# plt.grid()
# %%
