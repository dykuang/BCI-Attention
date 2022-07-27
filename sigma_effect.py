'''
Poke around sigma's effect with saved models
'''
#%%
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from logging import raiseExceptions
from tensorflow.keras import layers
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import seaborn as sns
import umap

class K_attention(layers.Layer):
    def __init__(self, **kwargs):
        super(K_attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        '''
        _,K_dim,_ = input_shape
        # self.r_sigma = tf.Variable(initial_value=0.0, trainable = True, dtype=tf.float32)
        self.r_sigma = self.add_weight(
                            shape=(1,),
                            initializer= tf.constant_initializer(value=0.01), 
                            constraint=lambda x: tf.clip_by_value(x, -0.1, 2.0),
                            trainable=True, name='r_sigma'
                            )

    def call(self, x, norm='euclidean', kernel='Gaussian', use_mask = False):
        _,size,f_dim = x.shape
        self_diff = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
        Dist = tf.norm(self_diff, ord = norm, axis=-1)
        Dist = tf.cast(Dist, dtype=tf.float32)
        # print('*'*40)
        # print(Dist.shape)
        # print('*'*40)
        if kernel == 'Gaussian':
            K = tf.math.exp(-Dist**2 * self.r_sigma) # need reduce mean here? scale adjustment?
        
        else:
            raiseExceptions('Kernel type not supported.')

        if use_mask:
            mask = np.zeros((size,size), dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask = tf.convert_to_tensor(mask)
            mask = tf.cast(mask[None,None, ...], tf.float32)
            K = K * mask

        return x + K @ x


from tensorflow.keras import Model, layers, losses, metrics

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm


from Modules import *
# try pretrained with layer initialization?
def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             optimizer = Adam, learning_rate = 1e-3,
             selected = 'KAM'):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))

    # block1       = layers.Lambda(lambda x: x[...,0])(input1)
    # block1       = R_corr(name = 'Rcorr')(block1)
    # block1       = ButterW(name='BW')(block1)
    # block1       = K_attention_alpha(name = 'Katt')(block1)
    # block1       = layers.Lambda(lambda x: x[...,None])(block1)


    ##################################################################
    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)

    # block1       = R_corr(name = 'Rcorr')(block1)

    block1       = layers.BatchNormalization(axis = -1, name='BN-1')(block1)  # normalization on channels or time
    block1       = layers.DepthwiseConv2D((1, Chans), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv')(block1)
    block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    block1       = layers.Activation('elu')(block1)

    block1       = layers.AveragePooling2D((2, 1))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = layers.SeparableConv2D(F2, (5, 1),
                                   use_bias = False, padding = 'same',
                                    name = 'SepConv-1')(block1)

    # block2       = R_corr(name = 'Rcorr')(block2)
    if selected != 'eegnet':
        block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
        if selected == 'KAM':
            block2       = K_attention(name = 'Katt')(block2)
        elif selected == 'KAMMH':
            block2       = K_attention_MH(num_heads=4, name='Katt')(block2)
        elif selected == 'qkv':
            block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.1, name='Katt')(block2, use_mask=True)
        block2       = layers.Lambda(lambda x: x[...,None,:])(block2)

    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
       
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

model = EEGNet(nb_classes = 3, Chans = 62, Samples = 200, 
               dropoutRate = 0.5, kernLength = 5, F1 = 8, 
               D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
               learning_rate = 1e-2,
               selected = 'KAM')
model.summary()



#%%
'''
Some helper functions
'''
def make_epoch(x, start=0, end=np.infty, steps=200, stride=None):
    '''
    x: (chns, time)
    '''
    epochs = []
    head = start
    tail = start+steps
    if stride is None:
        stride = steps

    while tail<min(x.shape[-1], end):
        epochs.append( x[:,head:tail] )
        head = head+stride
        tail = tail+stride


    epochs  = np.array(epochs )  
    epochs  = zscore(epochs , axis=-1)
    epochs  = epochs.transpose((0,2,1))

    return epochs

def get_track(model, f_model, sigma_grid, epoched_trials):
    '''
    epoched_trials: list of epoched array, each element is from one trial
    summary: [trials][0:label,1:feature,2:Jac]
    '''
    summary = []
    for trial in epoched_trials:
        pred_track = []
        feature_track = []
        Jac_track = []

        for run_sigma in sigma_grid:
            model.get_layer('Katt').set_weights(np.array([[run_sigma]]))
            f_model.get_layer('Katt').set_weights(np.array([[run_sigma]]))
            pred_track.append( np.argmax(model.predict(trial[...,None]), axis=-1) )
            feature_track.append( f_model.predict(trial[...,None]) )  
            Jac = []
            for i in range(3):
                with tf.GradientTape() as tape:
                    x = f_model.get_layer('Katt').weights
                    tape.watch(x)
                    y = f_model(trial[...,None])[:,i]
                Jac.append(tape.jacobian(y, x)[0]) 
            Jac_track.append(Jac)

        summary.append([np.array(pred_track), 
                        np.array(feature_track), 
                        np.array(Jac_track)]) 

    return summary

#%%
'''
load particular trials data and normalize
'''
nn_token = 'kanet_v1'  # key words used for saving checkpoint during training 
ckpt_key = 'K_v1' # folder name used for putting saved checkpoint during training 
subject = '01'
raw = loadmat('/mnt/HDD/Datasets/SEED/Preprocessed_EEG/1_20131027.mat')
epoched_trials = []
trial_keys = list(raw.keys())[3:]
for i in range(3):
    epoched_trials.append(make_epoch(raw[trial_keys[i]]) )


#%%
'''
Getting a visualization of distribution of sigma per subject
'''
fold = 0
Sigmas = []
for i in range(1,16):
    temp = []
    for j in range(5):
        weight_path = os.path.join('/mnt/HDD/Datasets/SEED/ckpt', ckpt_key, 
                                'S{:02d}_checkpoint_{}_62chns_fold{}'.format(i, nn_token, j))
        model.load_weights(weight_path )
        temp.append( model.get_layer('Katt').get_weights()[0] )
    Sigmas.append(temp)

Sigmas = np.array(Sigmas)

clist = sns.color_palette('tab10') + sns.color_palette('Set2')[:-1]
plt.figure(figsize=(6,4))
for i in range(15):
    plt.scatter((i+1)*np.ones(5), Sigmas[i], color=clist[i])
plt.scatter(1, Sigmas[0][fold], color='r',s=72, marker='o')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Subject',fontsize = 14)
plt.ylabel(r'$\alpha$', fontsize=18)


# %%
'''
loading weights
'''

# fold = 0
weight_path = os.path.join('/mnt/HDD/Datasets/SEED/ckpt', ckpt_key, 
                           'S{}_checkpoint_{}_62chns_fold{}'.format(subject, 
                           nn_token, fold))
# weight_path = os.path.join('/mnt/HDD/Datasets/SEED/ckpt', 
#                            'S{}_checkpoint_{}_62chns_fold{}'.format(subject, 
#                            nn_token, fold))
model.load_weights(weight_path )
learned_sigma = model.get_layer('Katt').get_weights()
f_model = Model(model.input, model.get_layer('dense').output)


#%%
'''
preparing sigma_grid
'''
sigma_grid = np.linspace(0, 0.1, 11)

baseline = get_track(model, f_model, learned_sigma[0], epoched_trials)
sigma_track = get_track(model, f_model, sigma_grid, epoched_trials)

#%%
'''
performance change along the track
all the data
'''
# path = '/mnt/HDD/Datasets/SEED'

# X = loadmat( os.path.join(path, 'S{}_E01.mat'.format(subject)) )['segs'].transpose([2,1,0])
# Y = loadmat( os.path.join(path, 'Label.mat')  )['seg_labels'][0]
# '''
# Normalize
# '''
# ll = [2,1,0]
# XX = zscore(X, axis=1)
# epoched_trials_all = [XX[np.where(Y==l)] for l in ll]

# baseline_all = get_track(model, f_model, learned_sigma[0], epoched_trials_all)
# sigma_track_all = get_track(model, f_model, sigma_grid, epoched_trials_all)
# run_acc_all = []
# for j in range(len(sigma_grid)):
#     run_acc_all.append( [np.sum(sigma_track_all[0][0][j]==ll[i])/len(sigma_track_all[i][0][j]) for i in range(3)] )

# run_acc_all = np.array(run_acc_all)
# overall_acc_all = (len(baseline_all[0][0][0])*run_acc_all[:,0] + len(baseline_all[1][0][0])*run_acc_all[:,1]
#                   + len(baseline_all[2][0][0])*run_acc_all[:,2]) / len(Y)

# plt.figure()
# plt.plot(sigma_grid, run_acc_all[:,0], 'd-',label='Positive')
# plt.plot(sigma_grid, run_acc_all[:,1], 'd-',label='Neural')
# plt.plot(sigma_grid, run_acc_all[:,2], 'd-',label='Negative')
# plt.plot(sigma_grid, overall_acc_all, 'd-',label='Overall')
# plt.axvline(x=learned_sigma[0], color='black', linestyle='--')
# plt.legend()
#%%
'''
performance change along the track
first 3 trials
'''
ll = [2,1,0]
base_acc = [np.sum(baseline[i][0][0]==ll[i])/len(baseline[i][0][0]) for i in range(3)]

run_acc = []
for j in range(len(sigma_grid)):
    run_acc.append( [np.sum(sigma_track[i][0][j]==ll[i])/len(sigma_track[i][0][j]) for i in range(3)] )

run_acc = np.array(run_acc)
overall_acc = (len(baseline[0][0][0])*run_acc[:,0] + len(baseline[1][0][0])*run_acc[:,1]
               + len(baseline[2][0][0])*run_acc[:,2]) / (len(baseline[0][0][0]) + len(baseline[1][0][0]) +len(baseline[2][0][0]))
plt.figure(figsize=(6,6))
plt.plot(sigma_grid, run_acc[:,0], 'd-',label='Positive')
plt.plot(sigma_grid, run_acc[:,1], 'd-',label='Neural')
plt.plot(sigma_grid, run_acc[:,2], 'd-',label='Negative')
plt.plot(sigma_grid, overall_acc, '--',label='Overall')
plt.axvline(x=learned_sigma[0], color='black', linestyle='--')
plt.ylabel('Acc',fontsize = 14)
plt.xlabel(r'$\alpha$',fontsize = 18)
plt.legend(loc = 'center right')

#%%
'''
umap

may not be neccessary in SEED case since the feature space is of dim 3
'''
base_data = np.concatenate([b[1][0] for b in baseline], axis=0)
base_label = np.concatenate([2*np.ones(235), np.ones(233), np.zeros(206)])
track_data = np.concatenate([b[1] for b in sigma_track], axis=1)

mapper = umap.UMAP(n_neighbors=100, n_components=2, metric='euclidean', 
                   spread=1.0, min_dist=0.2, local_connectivity=1.0,
                   output_metric='euclidean', init='spectral', 
                   densmap=False, random_state = 12345)


embeded_o = mapper.fit_transform(base_data)
sns.scatterplot(x = embeded_o[:,0], y = embeded_o[:,1], hue = base_label)

embeded_track =[mapper.transform(f) for f in track_data]
plt.figure()
sns.scatterplot(x = embeded_track[0][:,0], y = embeded_track[0][:,1], hue = base_label)


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
coord_data = np.concatenate([t for t in embeded_track],axis=0)
embeded_df = pd.DataFrame(data=coord_data, columns=['UMAP-x', 'UMAP-y'])
embeded_df['alpha'] = np.repeat(sigma_grid,674)
embeded_df['label'] = np.tile(['Postive']*235+['Neural']*233+['Negative']*206, 11)

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
'''
plotly 3d static
'''
# import plotly.graph_objects as go
# traces = []
# data = 
# c = ['b', 'g', 'k']

# for x, y, z in data:
#     trace = go.Scatter3d(
#         x=x, y=z, z=y,
#         mode="lines",
#         # hovertext=text,
#         # hoverinfo="text",
#         line=dict(
#             # color=c[i],
#             cmin=0.0,
#             cmid=0.5,
#             cmax=1.0,
#             cauto=False,
#             colorscale="RdBu",
#             colorbar=dict(),
#             width=2.5,
#         ),
#         opacity=1.0,
#     )
#     traces.append(trace)

# fig = go.Figure(data=traces)
# fig.update_layout(
#     width=800,
#     height=600,
#     scene=dict(
#         aspectratio = dict( x=0.5, y=1.25, z=0.5 ),
#         yaxis_title=r"$\alpha",
#         xaxis_title="UMAP-X",
#         zaxis_title="UMAP-Y",
#     ),
#     scene_camera=dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
#     autosize=False,
#     showlegend=False,
# )
# fig_widget = go.FigureWidget(fig)
# # https://plotly.com/python/renderers/
# fig_widget.show(renderer="browser")





# %%
'''
derivative wrt sigma

be aware that tape.gradient takes the sum as output
use jacobian instead
'''
# f_model.get_layer('Katt').set_weights(learned_sigma)
# Jac = []
# for i in range(3):
#     with tf.GradientTape() as tape:
#         x = f_model.get_layer('Katt').weights
#         tape.watch(x)
#         y =  f_model(epoched_trials[0][...,None])[:,i]
#     Jac.append(tape.jacobian(y, x)[0])
# 
# 

# violin plot on baseline
# sns.violinplot(data=baseline[0][-1][0].transpose()[0], palette="light:g", inner="points", orient="h")
plt.rc('font', size=15) 
# hitogram-base
plt.figure()
plt.hist(baseline[0][-1][0][0], alpha = 0.4, label='Postive')
plt.hist(baseline[0][-1][0][1], alpha = 0.4, label='Neutral')
plt.hist(baseline[0][-1][0][2], alpha = 0.4, label='Negative')
plt.xlabel(r'$\frac{\partial f(x, \alpha)}{\partial \alpha} \vert_{x}$',
           fontsize = 24)
plt.legend(loc='upper left', prop={'size': 16})

#violin plot on sigma track
# colors = ['light:b', 
#           sns.light_palette("orange", n_colors=len(sigma_grid)), 
#          'light:g']
legends = ['Positive', 'Neutral', 'Negative']
for i in range(3):
    plt.figure()
    dd = sigma_track[0][-1][:,i,:].transpose()[0]
    sns.violinplot(data=dd, 
                   palette=sns.light_palette(clist[i], n_colors=len(sigma_grid)), 
                   inner=None, orient="h")
    plt.plot(np.mean(dd, axis=0), np.arange(len(sigma_grid)), 'rd--')
    plt.ylabel(r'$\alpha$', fontsize=18)
    plt.yticks(ticks=np.arange(len(sigma_grid)), labels=sigma_grid)
    plt.xlim([-125, 125])
    plt.xlabel(r'$\frac{\partial f(x, \alpha)}{\partial \alpha}\vert_{x}$', 
               fontsize = 24)
    plt.text(60, 10, legends[i], fontsize = 20)
    if i == 2:
        plt.text(50, 3.8, r'Learned $\alpha$', fontsize = 20)
    else:
        plt.text(-100, 3.8, r'Learned $\alpha$', fontsize = 20)

    plt.axhline(y=100*learned_sigma[0], xmin = -125, xmax = 125, color='k', ls='--')

    # plt.xlabel()
  

#%%
'''
Extracting channel attention weights for compared models:
eegnet, qkv, kanet_lb0, kanet_lbn, kanet_mh, se, cbam
'''
from Models import CANet
model_compared = ['eegnet', 'qkv','kanet_v3', 'kanet_v1', 'kanetMH_v1', 'SE', 'CBAM']
model_summary_folder = ['baseline', 'qkv', 'K_v3', 'K_v1', 'K_MH_v1', 'SE', 'CBAM']
model_spec = ['eegnet', 'qkv', 'KAM', 'KAM', 'KAMMH']
Collect = []
for i in range(7):
    if i < 5:
        model = EEGNet(nb_classes = 3, Chans = 62, Samples = 200, 
                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                learning_rate = 1e-2, selected=model_spec[i])
    elif i == 5:
        model = CANet(nb_classes = 3, Chans = 62, Samples = 200, attention_module = 'se_block',
                       dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                       D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                       learning_rate = 1e-2)
    elif i == 6:
        model = CANet(nb_classes = 3, Chans = 62, Samples = 200, attention_module = 'cbam_block',
                       dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                       D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                       learning_rate = 1e-2)
    W_list = []
    for fld  in range(5):
        weight_path = os.path.join('/mnt/HDD/Datasets/SEED/ckpt', model_summary_folder[i], 
                                'S{}_checkpoint_{}_62chns_fold{}'.format(subject, model_compared[i], fld))
        model.load_weights(weight_path )
        W = model.get_layer('DepthConv').weights
        W_list.append(W[0][0].numpy())
    W_list = np.concatenate(W_list, axis=-1)
    Collect.append(W_list)


## Create target array to save, after normalization

CC = np.array(Collect)[...,0,::2]
# savemat('/mnt/HDD/Datasets/SEED/benchmark_summary/ATT_DK_7models.mat', {'CM':CC})
        
# %%
'''
Load models for compare
'''
fold = 0
cpr_models = []
subject_selected = 1
for i, nn_token in enumerate(model_compared):
    if i < 5:
        model = EEGNet(nb_classes = 3, Chans = 62, Samples = 200, 
                dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                learning_rate = 1e-2, selected=model_spec[i])
    elif i == 5:
        model = CANet(nb_classes = 3, Chans = 62, Samples = 200, attention_module = 'se_block',
                       dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                       D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                       learning_rate = 1e-2)
    elif i == 6:
        model = CANet(nb_classes = 3, Chans = 62, Samples = 200, attention_module = 'cbam_block',
                       dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                       D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                       learning_rate = 1e-2)

    weight_path = os.path.join('/mnt/HDD/Datasets/SEED/ckpt', model_summary_folder[i], 
                            'S{:02d}_checkpoint_{}_62chns_fold{}'.format(subject_selected, nn_token, fold))
    model.load_weights(weight_path )
    cpr_models.append(model)


data_path = '/mnt/HDD/Datasets/SEED'
X = loadmat( os.path.join(data_path, 'S{:02d}_E01.mat'.format(subject_selected)) )['segs'].transpose([2,1,0])
chns = np.arange(62)
X = zscore(X[...,chns], axis=1)
chns_token = '{:02d}'.format(len(chns))
Y = loadmat( os.path.join(data_path, 'Label.mat') )['seg_labels'][0]+1

#%%
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
for i in [0,3,5]: #['eegnet', 'qkv','kanet_v3', 'kanet_v1', 'kanetMH_v1', 'SE', 'CBAM']
    temp_correct, _ = get_CM_idx(cpr_models[i], X, Y)
    Correct_list.append(temp_correct)

def common(lst1, lst2): 
    return list(set(lst1) & set(lst2))

Common_idx = []
for j in range(3): #j for labels
    a = Correct_list[0][j].copy()
    for i in range(1,3): #i for models
        a = common(a, Correct_list[i][j])
    Common_idx.append(a)


#%%
'''
Morphing between curves belonging to different labels
'''
def morphed_curve(cA, cB, grid=[0, 0.25, 0.5, 0.75, 1.0]):
    track = []
    for _h in grid:
        track.append((1-_h)*cA + _h*cB)
    return track


segs_for_morph = np.array(X[[c[1] for c in Common_idx]])


from itertools import combinations

interp_grid = np.linspace(0,1,101)
trackDict = dict.fromkeys(['eegnet', 'SE', 'kanet_v1'])
crossDict = dict.fromkeys(['eegnet', 'SE', 'kanet_v1'])
for _k in trackDict.keys():
    temp_track = []
    temp_cross = []
    i = model_compared.index(_k)
    for _p in combinations(np.arange(len(segs_for_morph)), 2):      
        track = morphed_curve(segs_for_morph[_p[0]], segs_for_morph[_p[1]], interp_grid)
        temp_track.append( cpr_models[i].predict(np.array(track)[...,None]) )
        temp_cross.append( (interp_grid[1]-interp_grid[0])*np.argmin( abs(temp_track[-1][:,_p[0]] - temp_track[-1][:,_p[1]]) )  )
    trackDict[model_compared[i]] = temp_track
    crossDict[model_compared[i]] = temp_cross

for i in trackDict['eegnet']:
    plt.figure()
    plt.plot(interp_grid, i, '+--')
    plt.legend(['Neg', 'Neu', 'Pos'])


#%%
def trig_abl_plot(cross_track):
    theta = np.array([90, 210, 330, 90])
    theta = theta*np.pi/180.0
    Data2Plot = [ [ cross_track[0], 1, cross_track[1]],  
                  [1, 1 - cross_track[0],  cross_track[2]],
                  [ 1 - cross_track[1], 1-cross_track[0], 1]
                 ]
    labels = ['Neu.', 'Neg.', 'Pos.'] #(1, 0, 2)
    fig_label = ['Neg.', 'Neu.', 'Pos.']
    
    colors = ['g', 'orange', 'cyan']
    plt.figure()
    for i in range(3):
        ax = plt.subplot(1,3,i+1, projection='polar')
        temp = np.append(Data2Plot[i], Data2Plot[i][0])
        ax.plot(theta, temp,colors[i])
        ax.fill(theta, temp,colors[i], alpha=0.3)
        ax.set_xticks(theta[:3])
        ax.set_xticklabels(labels, y=0.1)
        ax.set_ylim([0, 1.0])
        ax.set_title('Origin-{} \n {:.02f}'.format(fig_label[i], np.sum(temp[:3])), color=colors[i], size=10)


trig_abl_plot(crossDict['eegnet'])
# %%
'''
Prediction Transition Curve (PTC)
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

trace_plot_on_hp(trackDict['eegnet'], attach_legend=False, title = 'EEGNet')
trace_plot_on_hp(trackDict['SE'], attach_legend=False, title = '+SE')
trace_plot_on_hp(trackDict['kanet_v1'], attach_legend=True, title = '+KAM')


# %%
