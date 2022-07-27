'''
Candidate models
'''
#%%
from Modules import SwinTransformer, PatchExtract, PatchEmbedding, PatchMerging
from tensorflow.keras import Model, layers, losses, metrics
import tensorflow_addons as tfa

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm

'''
Regular CNN
'''

def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5,
                optimizer = Adam, learning_rate = 1e-3):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = layers.Input((Samples, Chans, 1))
    block1       = layers.Conv2D(8, (5, 1), 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = layers.Conv2D(8, (1, Chans),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = layers.Activation('elu')(block1)
    block1       = layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block1)
    block1       = layers.Dropout(dropoutRate)(block1)
  
    block2       = layers.Conv2D(16, (5, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block2)
    block2       = layers.Dropout(dropoutRate)(block2)
    
    block3       = layers.Conv2D(32, (5, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3       = layers.Activation('elu')(block3)
    block3       = layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block3)
    block3       = layers.Dropout(dropoutRate)(block3)
    
    block4       = layers.Conv2D(64, (5, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    
    block4       = layers.Lambda(lambda x: x[...,0,:])(block4)
    block4       = layers.Permute((2,1))(block4)
    block4       = K_attention(name = 'Katt')(block4)
    # block4       = K_attention_ex(name = 'Katt_ex', use_margin=True, offdiag_mask=True)(block4)
    block4       = layers.Permute((2,1))(block4)
    block4       = layers.Lambda(lambda x: x[...,None,:])(block4)
    
    block4       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4       = layers.Activation('elu')(block4)
    block4       = layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1))(block4)
    block4       = layers.Dropout(dropoutRate)(block4)
    
    flatten      = layers.Flatten()(block4)
    
    dense        = layers.Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = layers.Activation('softmax')(dense)
    
    Mymodel      = Model(inputs=input_main, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

'''
EEGNET
'''

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             optimizer = Adam, learning_rate = 1e-3):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))

    ##################################################################
    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)
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

'''
SEER
'''
def fork_merge(x_in, shape, 
               depth_rate,
               depth_multiplier, 
               kernel_num, kernel_len, dropout_rate=0.2, 
               normrate_head = 0.5,
               dep_activation='tanh',
               sep_activation = 'linear',
               merge_style = 'A',
               _label = None):
    
    #The F branch ==================================
    x = layers.DepthwiseConv2D((1, shape[0]), strides=(1, 1), padding="valid",
                    depth_multiplier=depth_multiplier,
                    data_format=None, dilation_rate=(1, 1), 
                    activation=None,  use_bias=False,
                    depthwise_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    depthwise_regularizer=None, bias_regularizer=None,
                    activity_regularizer=None, depthwise_constraint=max_norm(normrate_head),
                    bias_constraint=None, name = 'F_Dep_{}'.format(_label) )(x_in)
    x = layers.BatchNormalization(momentum=0.9, axis=-1)(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.Activation(dep_activation)(x)
    
    # x = Permute((-2,-1))(x)
    # x = Reshape((-1, in_shape[-1]))(x)
    x = layers.Lambda(lambda y: y[:, :, 0, :])(x)
    
    x = layers.SeparableConv1D( kernel_num, kernel_size=kernel_len, strides=1, padding="same",
                         data_format=None, dilation_rate=1, depth_multiplier= 1,
                         activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None, name = 'F_Sep_{}'.format(_label)
                         )(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Dropout(droprate)(x)
    
    
    # The C branch =======================================================
    xt = layers.Permute((1,3,2))(x_in)
    # xt = GaussianDropout(0.5)(xt)
    xt = layers.DepthwiseConv2D((1, shape[-1]), strides=(1, 1), padding="valid",
                         depth_multiplier=depth_multiplier,
                         activation=None,  use_bias=False,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_constraint=max_norm(normrate_head),
                         name = 'C_Dep_{}'.format(_label))(xt)
    xt = layers.BatchNormalization(momentum=0.9,axis=-1)(xt)
    xt = layers.SpatialDropout2D(dropout_rate)(xt)
    xt = layers.Activation(dep_activation)(xt)
    xt = layers.Lambda(lambda y: y[:, :, 0, :])(xt)
    
    xt = layers.SeparableConv1D( kernel_num, kernel_size=kernel_len, strides=1, padding="same",
                         data_format=None, dilation_rate=1, depth_multiplier=1,
                         activation=sep_activation, use_bias=False, depthwise_initializer="glorot_normal",
                         pointwise_initializer="glorot_uniform", bias_initializer="zeros",
                         depthwise_regularizer=None, pointwise_regularizer=None,
                         bias_regularizer=None,  activity_regularizer=None,
                         depthwise_constraint=None, pointwise_constraint=None,
                         bias_constraint=None, name = 'C_Sep_{}'.format(_label)
                         )(xt)
    
    # xt = Dropout(droprate)(xt)
    # Merge two branches =====================================================
    
    if merge_style == 'A':
        # x = attach_attention_module(x, 'se_block', ratio=2)   
        x = layers.Add()([x, xt])  

    elif merge_style == 'C':
        x = layers.Concatenate()([x, xt])
    elif merge_style == 'M':
        x = layers.Multiply()([x, xt])
    elif merge_style == 'W':
        x = layers.Weight()([x, xt])
    elif merge_style == 'CW':
        x = layers.C_Weight()([x, xt])
    
    return x

#%%
from Modules import WaveletDeconvolution, attach_attention_module

def TFCNet_multiWD(in_shape, num_classes, 
                   dep_activation = 'tanh', sep_activation = 'linear',
                   merge_style = 'A', use_WD = False,
                   WDspec_list = [[8, 5, 1]], # Number, len, strides
                   depth_multiplier = 1, depth_rate=1, #WD_channels = 16,
                   merge_kernel_num = 8, merge_kernel_len = 5,
                   num_filters_list = [16, 32], kernel_size_list=[5,5],
                   strides_for_pool=[5,5],
                   optimizer=Adam, learning_rate=1e-3, 
                   droprate=0.5, spatial_droprate=0.2,
                   normrate_head=1.0, normrate_dense = 0.5):
    
    x_in = layers.Input(shape = in_shape, name = 'input')
    # x_wd = GaussianDropout(0.1)(x_in)
    # x_wd = Dropout(0.5)(x_in)
    # x_wd = WaveletDeconvolution(WD_spec[0], kernel_length=WD_spec[1], strides=WD_spec[2], use_bias = True,
    #                           padding='same', data_format='channels_last', name='WD-1')(x_wd)
    # x_wd = BatchNormalization(axis=1)(x_in)
    if use_WD:
    
        x_wd = WaveletDeconvolution(WDspec_list[0][0], kernel_length=WDspec_list[0][1], strides=WDspec_list[0][2], 
                                    use_bias = False,
                                    padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_in)
    # x_wd_1 = WaveletDeconvolution(WDspec_list[0][0], kernel_length=2*WDspec_list[0][1], strides=WDspec_list[0][2], 
    #                               use_bias = False,
    #                               padding='same', data_format='channels_last', name='WD-{}-1'.format(0))(x_in)
    # x_wd_2 = WaveletDeconvolution(WDspec_list[0][0], kernel_length=4*WDspec_list[0][1], strides=WDspec_list[0][2], 
    #                               use_bias = False,
    #                               padding='same', data_format='channels_last', name='WD-{}-2'.format(0))(x_in)
    
    # x_wd = concatenate([x_wd, x_wd_1, x_wd_2], axis=-2)
    
    # # x_wd = BatchNormalization(axis=1)(x_wd) # which dimension to normalize?
    # x_wd = SpatialDropout2D(spatial_droprate)(x_wd)
    else:
        x_wd = layers.Lambda(lambda x: x[...,None])(x_in)
        x_wd = layers.Conv2D(WDspec_list[0][0], kernel_size=(WDspec_list[0][1],1), strides=WDspec_list[0][2], 
                      kernel_initializer = 'glorot_normal',
                      # groups = 1,
                      kernel_constraint=max_norm(normrate_head),
                      use_bias = False,
                      padding='same', data_format='channels_last', name='WD-{}'.format(0))(x_wd)
        x_wd = layers.BatchNormalization(momentum=0.9, axis=-1)(x_wd)
        x_wd = layers.Permute((1,3,2))(x_wd)
   
    
    x = fork_merge(x_wd, (WDspec_list[0][0], in_shape[-1]), 
                   depth_rate,
                   depth_multiplier, 
                   merge_kernel_num, merge_kernel_len, dropout_rate=spatial_droprate, 
                   normrate_head = normrate_head,
                   dep_activation = dep_activation,
                   sep_activation = sep_activation,
                   merge_style = merge_style,
                   _label = 0)
    x = layers.BatchNormalization(momentum=0.9,axis=-1)(x)
    # x = Activation('elu')(x)
    # x = GaussianDropout(droprate)(x)
    x = layers.SpatialDropout1D(spatial_droprate)(x)
    
    for i, spec in enumerate(WDspec_list[1:]):
        if use_WD:
            x = WaveletDeconvolution(spec[0], kernel_length=spec[1], strides=spec[2], use_bias = False,
                                      padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
        else:
            x = layers.Lambda(lambda x: x[...,None])(x)
            x = layers.Conv2D(spec[0], kernel_size=(spec[1], 1) , strides=spec[2], use_bias = False,
                       kernel_initializer = 'glorot_normal',
                       # groups = 1,
                       kernel_constraint=max_norm(normrate_head),
                       padding='same', data_format='channels_last', name='WD-{}'.format(i+1))(x)
            x = layers.BatchNormalization(momentum=0.9,axis=-1)(x)
            x = layers.Permute((1,3,2))(x)
            
        x = layers.SpatialDropout2D(spatial_droprate)(x)  
        
        x = fork_merge(x, (spec[0], merge_kernel_num), 
                       depth_rate,
                       depth_multiplier, 
                       merge_kernel_num, merge_kernel_len, 
                       dropout_rate=spatial_droprate, 
                       normrate_head = normrate_head,
                       dep_activation=dep_activation,
                       sep_activation = sep_activation,
                       _label = i+1)
        x = layers.BatchNormalization(momentum=0.9, axis=-1)(x)
        # x = Activation('elu')(x)
        # x = GaussianDropout(spatial_droprate)(x)
        # x = SpatialDropout1D(spatial_droprate)(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = GaussianDropout(spatial_droprate)(x)
    x = layers.SpatialDropout1D(spatial_droprate)(x)
    for i in range(len(strides_for_pool)):
        # x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], padding='same',
        #                     # depthwise_constraint=max_norm(normrate_head/2),
        #                     # activity_regularizer=l1(1e-5),
        #                     activation=None, use_bias = False, name='sepconv-{}'.format(i))(x)
        # # x = BatchNormalization(axis=-1)(x)
        # x = Activation('elu')(x)
        # x = SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], strides = strides_for_pool[i],
        #                     # depthwise_constraint=max_norm(normrate_head/2),
        #                     # activity_regularizer=l1(1e-5),
        #                     padding='same',
        #                     activation=None, use_bias = False, name='pooling-{}'.format(i))(x)
        # x = BatchNormalization(axis=-1)(x)
        # x = SpatialDropout1D(spatial_droprate)(x)
        # x = Activation('elu')(x)
        
        
        # try pooling
        x = layers.SeparableConv1D(num_filters_list[i], kernel_size=kernel_size_list[i], strides = 1,
                            # depthwise_constraint=max_norm(normrate_head/2),
                            # activity_regularizer=l1(1e-5),
                            depthwise_initializer="glorot_normal",
                            pointwise_initializer="glorot_uniform",
                            activation=None, use_bias = False)(x)
        x = layers.BatchNormalization(momentum=0.9, axis=-1)(x)
        x = layers.SpatialDropout1D(spatial_droprate)(x)
        x = layers.Activation('elu')(x)
        x = layers.MaxPooling1D(strides_for_pool[i], name='pooling-{}'.format(i))(x)

    # x = Lambda(lambda x: x[...,None,:])(x)
    # x = attach_attention_module(x, 'se_block', ratio=2)    
    # x = Lambda(lambda x: x[...,0,:])(x)

    # x = SeparableConv1D(num_filters_list[2], kernel_size=1, use_bias = True, name='sepconv-3')(x)
    # x = BatchNormalization(axis=1)(x)
    # x = Activation('elu')(x)
    
    x = layers.Dropout(droprate)(x)
    
    x = layers.Flatten(name = 'flatten')(x)
    # x = GlobalAveragePooling1D(name = 'flatten')(x)
    
    # x = Dense(32, name='feature', activation = 'elu')(x)
    
    x = layers.Dense(num_classes, name = 'dense_last', kernel_constraint = max_norm(normrate_dense) )(x)
    
    softmax = layers.Activation('softmax', name = 'softmax')(x)
    
    Mymodel = Model(x_in, softmax)
    Mymodel.compile(
                    loss='categorical_crossentropy', 
                    # loss = damped_log_loss,
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    
    return Mymodel


'''
Modified EEGNet with correlation/connectivity matrix multiplied on the right
or attention matrix generated from kernel multiplied on the left 
'''
from Modules import *
def KANet(nb_classes, Chans = 64, Samples = 128, 
           dropoutRate = 0.5, kernLength = 64, F1 = 8, 
           D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
           optimizer = Adam, learning_rate = 1e-3):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))

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
    block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    block2       = K_attention(name = 'Katt')(block2)
    # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    #                             name='Katt')(block2,kernel='Linear',use_mask=False)
    # block2       = K_attention_MH(num_heads=4, name='Katt')(block2)
    # block2       = K_attention_ex(name = 'Katt')(block2)
    # block2       = KAM_R(name = 'C2A_NNR')(block2, offdiag_mask=True, use_mask=False)

    block2       = layers.Lambda(lambda x: x[...,None,:])(block2)
    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    # block2       = dropoutType(dropoutRate)(block2)

    # block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    # block2       = K_attention(name = 'Katt')(block2)
    # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    #                             name='Katt')(block2,kernel='Linear',use_mask=False)
    # block2       = K_attention_MH(num_heads=4, name='Katt')(block2)
    # block2       = K_attention_ex(name = 'Katt')(block2)
    # block2       = KAM_R(name = 'C2A_NNR')(block2, offdiag_mask=True, use_mask=False)
    # block2       = layers.Lambda(lambda x: x[...,None,:])(block2)
        
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel


'''
An inserted transformer that converts deep Gram matrix
per batch to attention matrix
'''
def MTNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             penalty_rate=1.0, mono_mode =0,
             optimizer = Adam, learning_rate = 1e-3):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1       = layers.Input(shape = (Samples, Chans, 1))

    # block1       = layers.Lambda(lambda x: x[...,0])(input1)
    # # block1       = Spatial_att(init=10.0, vmin=-.01, vmax=30, channel_wise=False, name='SaM')(block1, Dist_M,offdiag_mask=True)
    # block1       = D2A(name='D2A')(block1, Dist_M, offdiag_mask = True)
    # # block1       = DA_mono(name='DA_mono')(block1, Dist_M, offdiag_mask = True)
    # # block1       = layers.LayerNormalization()(block1)
    # block1       = layers.Lambda(lambda x: x[...,None])(block1)
    # # block1       = layers.BatchNormalization(axis = -1)(block1)

    ##################################################################
    # block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
    #                                use_bias = False, name = 'Conv2D')(block1)
    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)

    # block1       = R_corr(name = 'Rcorr', corr_passed=np.zeros((8,8)).astype(np.float32))(block1)
    # block1       = R_corr(name = 'Rcorr', corr_passed=None)(block1)
    # block1       = layers.Permute((3,1,2))(block1)
    # block1       = Spatial_att(init=0.01, vmin=-2, vmax=2,name='SaM')(block1, Dist_M,offdiag_mask=True)
    # block1       = layers.Permute((2,3,1))(block1)
    
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
    block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    #block2       = layers.Permute((2,1))(block2)
    # block2       = K_attention(name = 'Katt')(block2, offdiag_mask=True, use_mask=False)
    # block2       = K_attention_NN(name = 'Katt')(block2, offdiag_mask=True, use_mask=False)
    # block2       = K_attention_MH(num_heads=2, name='Katt')(block2)
    # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
                                #  name='Katt')(block2,kernel='Linear',use_mask=False)
    # block2       = K_attention_ex(name = 'Katt', use_margin=True, offdiag_mask=True)(block2)
    block2, penalty  = FC_mono(name = 'att_mono', penalty_rate=penalty_rate, mono_mode=mono_mode)(block2, offdiag_mask=True, use_mask=False)
    #block2       = layers.Permute((2,1))(block2)
    block2       = layers.Lambda(lambda x: x[...,None,:])(block2)

    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    # block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    # # block2       = K_attention(name = 'Katt')(block2, offdiag_mask=True, use_mask=False)
    # # block2       = K_attention_NN(name = 'Katt')(block2, offdiag_mask=True, use_mask=False)
    # # block2       = K_attention_MH(num_heads=2, name='Katt')(block2)
    # # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    #                             #  name='Katt')(block2,kernel='Linear',use_mask=False)
    # # block2       = K_attention_ex(name = 'Katt', use_margin=True, offdiag_mask=True)(block2)
    # block2, penalty  = K_attention_mono(name = 'att_mono')(block2, offdiag_mask=True, use_mask=False, mono_mode = 1)
    # block2       = layers.Lambda(lambda x: x[...,None,:])(block2)
       
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)

    Mymodel.add_loss(penalty)
    Mymodel.add_metric(penalty, name='mono_penalty')
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(learning_rate=learning_rate))
    
    return Mymodel

'''
Attentional modules such as SE and CBAM inserted in EEGNet
'''
def CANet(nb_classes, Chans = 64, Samples = 128, attention_module = 'se_block',
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             optimizer = Adam, learning_rate = 1e-3):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))

    ##################################################################
    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)

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

    block2       = attach_attention_module(block2, attention_module)

    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
       
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(nb_classes, name = 'last_dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(learning_rate=learning_rate))
    
    return Mymodel


'''
The easy QKV type attentional module inserted in EEGNet
'''
def QKVNet(nb_classes, Chans = 64, Samples = 128, 
           dropoutRate = 0.5, kernLength = 64, F1 = 8, 
           D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
           optimizer = Adam, learning_rate = 1e-3):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))

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
    block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    # block2       = K_attention(name = 'Katt')(block2)
    block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
                                 name='QKV-att')(block2,kernel='Linear',use_mask=False)
    # block2       = K_attention_MH(num_heads=4, name='Katt')(block2)
    # block2       = K_attention_ex(name = 'Katt')(block2)
    # block2       = KAM_R(name = 'C2A_NNR')(block2, offdiag_mask=True, use_mask=False)

    block2       = layers.Lambda(lambda x: x[...,None,:])(block2)
    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    # block2       = dropoutType(dropoutRate)(block2)

    # block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    # block2       = K_attention(name = 'Katt')(block2)
    # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    #                             name='Katt')(block2,kernel='Linear',use_mask=False)
    # block2       = K_attention_MH(num_heads=4, name='Katt')(block2)
    # block2       = K_attention_ex(name = 'Katt')(block2)
    # block2       = KAM_R(name = 'C2A_NNR')(block2, offdiag_mask=True, use_mask=False)
    # block2       = layers.Lambda(lambda x: x[...,None,:])(block2)
        
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel
