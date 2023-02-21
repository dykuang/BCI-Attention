#%%
from random import sample
import  numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from Utils import scores
from Models import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from scipy.io import loadmat

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject', help='subject index')
parser.add_argument('--expType', help='subject index')
parser.add_argument('--nn_token', help='subject index')
parser.add_argument('--valMode', help='subject index')  # 'random' or 'fix'

args = parser.parse_args()

subject = '{:02d}'.format( int(args.subject) )
exp_type = int(args.expType)
nn_token = args.nn_token
val_mode = args.valMode

#%%
# subject = 2
# exp_type = 0 # 0: H/L Valence  1: H/L arousal 2: H/L V/A

if exp_type<2:
    num_class = 2
elif exp_type == 2:
    num_class = 4
else:
    num_class = None
    print('Invalid number of classes. ')
path = '/mnt/HDD/Datasets/DEAP/s{:02d}.mat'.format( int(subject) )
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

# 4 class label on valence & arousal
# 0: HVHA, 1: HVLA, 2:LVHA, 3: LVLA
label_VA_TF = -1* np.ones(len(label_A))
label_VA_TF[label_V_TF & label_A_TF] = 0
label_VA_TF[np.logical_and(label_V_TF, np.logical_not(label_A_TF))] = 1
label_VA_TF[np.logical_and(np.logical_not(label_V_TF), label_A_TF)] = 2
label_VA_TF[np.logical_not(label_V_TF) & np.logical_not(label_A_TF)] = 3
label_VA = to_categorical(label_VA_TF)

#%%
'''
write a generator
need to make sure providing samples w different labels at almost equal probability?
'''
def sample_generator(XData, YLabel, seg_len=128, window=(0,2000), batchsize=128):
    '''
    XData: data to slice from
    YLabel: one-hot label
    '''
    num_cls = YLabel.shape[1]
    X = np.zeros((batchsize, seg_len, 32))
    Y = np.zeros((batchsize, num_cls))
    
    label_pos = np.argmax(YLabel, axis=1)
    
    Type_idx = [np.where(label_pos == i)[0] for i in range(num_cls)]

    # T_idx = np.where(label_pos == 1)[0]
    # F_idx = np.where(label_pos == 0)[0]

    while True:
        start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
        for i, idx in enumerate(start_idx):
            dice = np.random.randint(0,num_cls)
            trial_num = np.random.randint(0, len(Type_idx[dice]), 1)
            selected = Type_idx[dice][trial_num]         
            # if dice:
            #     trial_num = np.random.randint(0, len(T_idx), 1)
            #     selected = T_idx[trial_num]

            # else:
            #     trial_num = np.random.randint(0, len(F_idx), 1)
            #     selected = F_idx[trial_num]
           
            X[i] = XData[selected, idx:idx+seg_len, :]
            Y[i] = YLabel[selected]
        
        yield X,  Y
    
 
def val_gen(XData, YLabel, seg_len=128, window=(0,2000), batchsize=128):
    num_cls = YLabel.shape[1]
    X = np.zeros((batchsize, seg_len, 32))
    Y = np.zeros((batchsize, num_cls))
    
    label_pos = np.argmax(YLabel, axis=1)
    
    Type_idx = [np.where(label_pos == i)[0] for i in range(num_cls)]

    start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
    for i, idx in enumerate(start_idx):
        dice = np.random.randint(0,num_cls)
        trial_num = np.random.randint(0, len(Type_idx[dice]), 1)
        selected = Type_idx[dice][trial_num]         
        
        X[i] = XData[selected, idx:idx+seg_len, :]
        Y[i] = YLabel[selected]
    
    return X,  Y


def make_segs(data, seg_len, stride):
    t_len = data.shape[1]
    segs = np.stack([data[:,i*stride:i*stride+seg_len,:] for i in range(t_len//stride) if i*stride+seg_len<=t_len], axis= 1)
    # print(segs.shape)
    return segs.reshape((-1, seg_len, data.shape[-1]))



#%% Building models
# from keras_swin_template import SwinTransformer, PatchExtract, PatchEmbedding, PatchMerging
from tensorflow.keras import Model, layers, losses, metrics
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm

# try pretrained with layer initialization?
# def EEGNet(nb_classes, Chans = 64, Samples = 128, 
#              dropoutRate = 0.5, kernLength = 64, F1 = 8, 
#              D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
#              optimizer = Adam, learning_rate = 1e-3):
   
#     if dropoutType == 'SpatialDropout2D':
#         dropoutType = layers.SpatialDropout2D
#     elif dropoutType == 'Dropout':
#         dropoutType = layers.Dropout
#     else:
#         raise ValueError('dropoutType must be one of SpatialDropout2D '
#                          'or Dropout, passed as a string.')
    
#     input1   = layers.Input(shape = (Samples, Chans, 1))

#     # block1       = layers.Lambda(lambda x: x[...,0])(input1)
#     # # block1       = ButterW(name='BW')(block1)
#     # block1       = layers.Permute((2,1))(block1)
#     # block1       = K_attention(name = 'Katt')(block1)
#     # block1       = layers.Permute((2,1))(block1)
#     # block1       = layers.Lambda(lambda x: x[...,None])(block1)


#     ##################################################################
#     # block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
#     #                                use_bias = False, name = 'Conv2D')(block1)
#     block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
#                                    use_bias = False, name = 'Conv2D')(input1)

#     # block1       = R_corr(name = 'Rcorr', corr_passed=np.zeros((8,8)).astype(np.float32))(block1)
#     # block1       = R_corr(name = 'Rcorr', corr_passed=None)(block1)
    
#     block1       = layers.BatchNormalization(axis = -1, name='BN-1')(block1)  # normalization on channels or time
#     block1       = layers.DepthwiseConv2D((1, Chans), use_bias = False, 
#                                    depth_multiplier = D,
#                                    depthwise_constraint = max_norm(1.),
#                                    name = 'DepthConv')(block1)
#     block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
#     block1       = layers.Activation('elu')(block1)

#     block1       = layers.AveragePooling2D((2, 1))(block1)
#     block1       = dropoutType(dropoutRate)(block1)
    
#     block2       = layers.SeparableConv2D(F2, (5, 1),
#                                    use_bias = False, padding = 'same',
#                                     name = 'SepConv-1')(block1)

#     # block2       = R_corr(name = 'Rcorr')(block2)
#     # block2       = attach_attention_module(block2, 'se_block')
#     # block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
#     # block2       = K_attention(name = 'Katt')(block2, use_mask=False)
#     # block2       = C_attention(name = 'Catt')(block2, use_mask=False)
#     # block2       = K_attention_MH(num_heads=2, name='Katt')(block2)
#     # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
#                                 #  name='Katt')(block2,kernel='Linear',use_mask=False)
#     # block2       = layers.Lambda(lambda x: x[...,None,:])(block2)

#     block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
#     block2       = layers.Activation('elu')(block2)
#     block2       = layers.AveragePooling2D((2, 1))(block2)
#     block2       = dropoutType(dropoutRate)(block2)
       
#     flatten      = layers.Flatten(name = 'flatten')(block2)
    
#     dense        = layers.Dense(nb_classes, name = 'last_dense', 
#                          kernel_constraint = max_norm(norm_rate))(flatten)
#     softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
#     Mymodel      = Model(inputs=input1, outputs=softmax)
    
#     Mymodel.compile(loss='categorical_crossentropy', 
#                     metrics=['accuracy'],
#                     optimizer=optimizer(learning_rate=learning_rate))
    
#     return Mymodel


#%%
# def Deapnet(nb_classes, Chans = 64, Samples = 256,
#                 dropoutRate = 0.2,
#                 input_shape = (256,1),
#                 penalty_rate=1.0, mono_mode ='ID',
#                 optimizer = Adam
#                 #learning_rate = 1e-3
#                 ):

#     #input = Input(shape=(Chans,1))
#     input = layers.Input(shape = (Samples, Chans))
#     block1 = layers.Conv1D(128, 3, activation='relu')(input)
#     block1 = layers.MaxPooling1D(pool_size=2)(block1)
#     block1 = layers.Dropout(dropoutRate)(block1)

#     block2 = layers.Conv1D(128, 3,  activation='relu')(block1)
#     #block2,penalty = FC_mono(name = 'att_mono', penalty_rate=1.0, mono_mode=mono_mode)(block2, offdiag_mask=True, use_mask=False)
#     block2 = layers.Dropout(dropoutRate)(block2)

#     block3 = layers.GRU(units = 256, return_sequences=True)(block2)
#     block3 = layers.Dropout(dropoutRate)(block3)

#     block4 = layers.GRU(units = 32)(block3)
#     block4 = layers.Dropout(dropoutRate)(block4)

#     flatten = layers.Flatten()(block4)

#     dense1 = layers.Dense(units = 128, activation='relu')(flatten)
#     block5 = layers.Dropout(dropoutRate)(dense1)

#     dense2 = layers.Dense(units = nb_classes)(block5)
#     softmax = layers.Activation('softmax')(dense2)

#     Mymodel  = Model(inputs=input, outputs=softmax)
    
#     # Mymodel.compile(loss='categorical_crossentropy', 
#     #                 metrics=['accuracy'],
#     #                 optimizer= 'adam')
    
#     return Mymodel

#%%
batchsize = 256
seg_len = 128
lr = 1e-3

# nn_token = 'EEGNet'
# nn_token = 'SE'
# nn_token = 'Mnt_ID'

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
elif nn_token == 'KAM':
    model = KANet(nb_classes = num_class, Chans = 32, Samples = seg_len, 
                  dropoutRate = 0.5, kernLength = 5, F1 = 8, 
                  D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                  optimizer = Adam, learning_rate = lr)    
elif nn_token == 'QKV':
    model = QKVNet(nb_classes = num_class, Chans = 32, Samples = seg_len, 
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
elif nn_token in ['SEER']:
    Params = {
        'shape': (seg_len, 32),
        'num classes': num_class,
        'depth_act': 'tanh',
        'sep_act': 'linear',
        'merge': 'A',
        'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
        'depth multiplier':1,
        'depth rate':1,
        'merge ker num': 8,
        'merge ker len': 5,
        'num_filters_list':[8, 8], 
        'kernel_size_list':[5,5],
        'strides_for_pool':[2,2],   
        'droprate':0.5, 
        'spatial droprate': 0.0,
        'normrate_head': 1.0, 
        'normrate_dense':0.25,
        'lr':lr
        }
    model = TFCNet_multiWD(Params['shape'], Params['num classes'], 
                           dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
                           merge_style = Params['merge'], use_WD = False,
                           WDspec_list = Params['WD_spec'], # Number, len, strides
                           depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
                           merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
                           num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
                           strides_for_pool=Params['strides_for_pool'],
                           learning_rate=Params['lr'], droprate=Params['droprate'], 
                           spatial_droprate=Params['spatial droprate'],
                           normrate_head=Params['normrate_head'], 
                           normrate_dense = Params['normrate_dense'])

else:
    assert('nn_token not recognized.')
# model = Deapnet(2, Chans=32, Samples = 128, dropoutRate=0.2)
# model = DeepConvNet(nb_classes = 2, Chans = 32, Samples = 128,
#                     dropoutRate = 0.25,
#                     optimizer = Adam, learning_rate = 1e-3)
# Params = {
#     'shape': (seg_len, 32),
#     'num classes': num_class,
#     'depth_act': 'tanh',
#     'sep_act': 'linear',
#     'merge': 'A',
#     'WD_spec' : [[8, 5, 1]]*1, # num, kernel length, stride
#     'depth multiplier':1,
#     'depth rate':1,
#     'merge ker num': 8,
#     'merge ker len': 5,
#     'num_filters_list':[8, 8], 
#     'kernel_size_list':[5,5],
#     'strides_for_pool':[2,2],   
#     'droprate':0.5, 
#     'spatial droprate': 0.0,
#     'normrate_head': 0.5, 
#     'normrate_dense':0.25,
#     }
# model = TFCNet_multiWD(Params['shape'], Params['num classes'], 
#                        dep_activation =  Params['depth_act'], sep_activation =  Params['sep_act'],
#                        merge_style = Params['merge'], use_WD = False,
#                        WDspec_list = Params['WD_spec'], # Number, len, strides
#                        depth_multiplier = Params['depth multiplier'], depth_rate=Params['depth rate'], 
#                        merge_kernel_num = Params['merge ker num'], merge_kernel_len = Params['merge ker len'],
#                        num_filters_list = Params['num_filters_list'], kernel_size_list=Params['kernel_size_list'],
#                        strides_for_pool=Params['strides_for_pool'],
#                        learning_rate=Params['lr'], droprate=Params['droprate'], 
#                        spatial_droprate=Params['spatial droprate'],
#                        normrate_head=Params['normrate_head'], 
#                        normrate_dense = Params['normrate_dense'])
# model.compile(loss='categorical_crossentropy', 
#                 metrics=['accuracy'],
#                 optimizer= Adam(learning_rate= lr))
model.summary()
model.save_weights('/mnt/HDD/Benchmarks/DEAP/model_ini.h5') # save the initial weights
#%% Pretrained with EEGNET

train_win = (0, 5000)
val_win = (5000, 6000)
val_batchsize = 1024
# test_win = (6000, data.shape[1])
Xtest= make_segs(data_N[:,6000:,:], seg_len, seg_len)

if exp_type == 0:
    train_gen = sample_generator(data_N, label_V, seg_len, train_win, batchsize=batchsize)
    # test_gen = sample_generator(data_N, label_V, seg_len, test_win, batchsize=batchsize)
    Ytest = np.repeat(label_V, Xtest.shape[0]//40, axis=0)
    
    if val_mode == 'random':
        Val = val_gen(data_N, label_V, seg_len, val_win, val_batchsize )
    elif val_mode == 'fix':
        Val_x = make_segs(data_N[:,5000:6000,:], seg_len, seg_len)
        Val_y = np.repeat(label_V, Val_x.shape[0]//40, axis=0)
        Val = (Val_x, Val_y)

elif exp_type == 1:
    train_gen = sample_generator(data_N, label_A, seg_len, train_win, batchsize=batchsize)
    # test_gen = sample_generator(data_N, label_A, seg_len,test_win,  batchsize=batchsize)
    Ytest = np.repeat(label_A, Xtest.shape[0]//40, axis=0)
    if val_mode == 'random':
        Val = val_gen(data_N, label_A, seg_len, val_win, val_batchsize )
    elif val_mode == 'fix':
        Val_x = make_segs(data_N[:,5000:6000,:], seg_len, seg_len)
        Val_y = np.repeat(label_A, Val_x.shape[0]//40, axis=0)
        Val = (Val_x, Val_y)

elif exp_type == 2:
    train_gen = sample_generator(data_N, label_VA, seg_len, train_win, batchsize=batchsize)
    Ytest = np.repeat(label_VA, Xtest.shape[0]//40, axis=0)
    if val_mode == 'random':
        Val = val_gen(data_N, label_VA, seg_len, val_win, val_batchsize )
    elif val_mode == 'fix':
        Val_x = make_segs(data_N[:,5000:6000,:], seg_len, seg_len)
        Val_y = np.repeat(label_VA, Val_x.shape[0]//40, axis=0)
        Val = (Val_x, Val_y)
else:
    print('experiement type not available yet')
    
#%%
from sklearn.metrics import confusion_matrix, accuracy_score
# training_history = []
count = 0 
summary = []
summary_weighted = []
ConM = []


for i in range(10):
    print('run {} started ...'.format(i))
    # Val = val_gen(data_N, label_V, seg_len, val_win, val_batchsize ) 
    cpt_path = '/mnt/HDD/Benchmarks/DEAP/ckpt/S{}_ckpt_{}_type{}_count{}'.format(subject, nn_token, exp_type, i)
        
    cpt = ModelCheckpoint(filepath=cpt_path,
                          save_weights_only=True,
                          monitor='val_accuracy',
                          mode='max',
                          save_best_only=True)

    model.load_weights('/mnt/HDD/Benchmarks/DEAP/model_ini.h5') # same initialization before fit
    hist = model.fit(train_gen,
                    epochs=10, 
                    steps_per_epoch= (train_win[1] - train_win[0]-seg_len)*40//batchsize,
                    validation_data = Val,
                    #  validation_steps= 2,
                    #  validation_split=0.2,
                    verbose=1,
                    callbacks=[cpt],
                    #  shuffle = True,
        #                  class_weight = weight_dict
                        )

        
    model.load_weights(cpt_path) #load best validation model
    pred = model.predict(Xtest)
    CM = confusion_matrix(np.argmax(Ytest,axis=1), np.argmax(pred, axis=1))
    # print(CM)
    
    a, b = scores(CM )
    # print(b)
    summary.append(a)
    summary_weighted.append(b)
    ConM.append( CM )
    count += 1
        
summary = np.array(summary)
summary_weighted = np.array(summary_weighted)

# print('mean: {}'.format(np.mean(summary, axis = 0)))
# print('std: {}'.format(np.std(summary, axis = 0))) 

with open('/mnt/HDD/Benchmarks/DEAP/type{}_{}_{}_history.txt'.format(exp_type, nn_token, val_mode), 'a') as file:
    file.write('\n' + '='*60 + '\n')
    file.write('{}\'s performance on Subject {}: \n'.format(nn_token, subject))

    file.write('\n')
    file.write('mean(W):  ')
    file.writelines(['{:.04f}    '.format(s) for s in np.mean(summary_weighted, axis = 0).reshape(-1)])
    file.write('\n')
    file.write('std(W):  ')
    file.writelines(['{:.04f}    '.format(s) for s in np.std(summary_weighted, axis = 0).reshape(-1)])
    file.write('\n')

np.save('/mnt/HDD/Benchmarks/DEAP/summary/S{}_{}_type{}_{}'.format(subject, nn_token, exp_type, val_mode), summary)
np.save('/mnt/HDD/Benchmarks/DEAP/summary/SW{}_{}_type{}_{}'.format(subject, nn_token, exp_type, val_mode), summary_weighted)
np.save('/mnt/HDD/Benchmarks/DEAP/summary/CM_S{}_{}_type{}_{}'.format(subject, nn_token, exp_type, val_mode), ConM)


# with plt.style.context('bmh'):
#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.plot(hist.history['loss'])
#     plt.plot(hist.history['val_loss'])
#     plt.title('Loss')
#     plt.subplot(1,2,2)
#     plt.plot(hist.history['accuracy'])
#     plt.plot(hist.history['val_accuracy'])
#     plt.title('Accuracy')


# print(accuracy_score(np.argmax(YTest_V, axis=1), np.argmax(pred, axis=1)))
# CM = confusion_matrix(np.argmax(YTest_V, axis=1), np.argmax(pred, axis=1))
# print(CM )
# %%
