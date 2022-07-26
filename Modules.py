'''
custom layers
'''

#%%

from logging import raiseExceptions
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.backend import bias_add
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


#####################################################################################
# Transformer type
######################################################################################

'''
Modifications needed:
window_size --> tuple (height, width)
shift_size --> tuple (height, width)
     y: height: 0
     x: width: 1
     image: height, width, channels
'''

'''
Wavelet decomposition layer

modified from https://github.com/haidark/WaveletDeconv/blob/master/WaveletDeconvolution.py
'''
import tensorflow.compat.v1.keras.backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import activations, initializers, regularizers, constraints
class Pos(constraints.Constraint):
    '''Constrain the weights to be strictly positive
    '''
    def __call__(self, p):
        p *= K.cast(p > 0., K.floatx())
        return p
class WaveletDeconvolution(layers.Layer):
    '''
    Deconvolutions of 1D signals using wavelets
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`  as a
    (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors with dimension 128).
    
    # Example
    ```python
        # apply a set of 5 wavelet deconv widthss to a sequence of 32 vectors with 10 timesteps
        model = Sequential()
        model.add(WaveletDeconvolution(5, padding='same', input_shape=(32, 10)))
        # now model.output_shape == (None, 32, 10, 5)
        # add a new conv2d on top
        model.add(Conv2D(64, 3, 3, padding='same'))
        # now model.output_shape == (None, 64, 10, 5)
    ```
    # Arguments
        nb_widths: Number of wavelet kernels to use
            (dimensionality of the output).
        kernel_length: The length of the wavelet kernels            
        init: Locked to didactic set of widths ([1, 2, 4, 8, 16, ...]) 
            name of initialization function for the weights of the layer
            (see [initializers](../initializers.md)),
            or alternatively, a function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            ( or alternatively, an elementwise function.)
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix
        bias_constraint: Constraint function applied to the bias vector
    
    # Input shape
        if data_format is 'channels_first' then
            3D tensor with shape: `(batch_samples, input_dim, steps)`.
        if data_format is 'channels_last' then
            3D tensor with shape: `(batch_samples, steps, input_dim)`.
        
    # Output shape
        if data_format is 'channels_first' then
            4D tensor with shape: `(batch_samples, input_dim, new_steps, nb_widths)`.
            `steps` value might have changed due to padding.
        if data_format is 'channels_last' then
            4D tensor with shape: `(batch_samples, new_steps, nb_widths, input_dim)`.
            `steps` value might have changed due to padding.
    '''
    
    def __init__(self, nb_widths, kernel_length=100,
                 init='uniform', activation='linear', weights=None,
                 padding='same', strides=1, data_format='channels_last', use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 input_shape=None, **kwargs):
        
        super(WaveletDeconvolution, self).__init__(**kwargs)

        if padding.lower() not in {'valid', 'same'}:
            raise Exception('Invalid border mode for WaveletDeconvolution:', padding)
        if data_format.lower() not in {'channels_first', 'channels_last'}:
            raise Exception('Invalid data format for WaveletDeconvolution:', data_format)
        self.nb_widths = nb_widths
        self.kernel_length = kernel_length
        self.init = self.didactic #initializers.get(init, data_format='channels_first')
        self.activation = activations.get(activation)
        self.padding = padding
        self.strides = strides

        self.subsample = (strides, 1)

        self.data_format = data_format.lower()

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = Pos()
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.initial_weights = weights
    
    def didactic(self, shape, name=None):
        x = 2**np.arange(shape).astype('float32')
        return K.variable(value=x, name=name)
        

    def build(self, input_shape):
        # get dimension and length of input
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]
            self.input_length = input_shape[2]
        else:
            self.input_dim = input_shape[2]
            self.input_length = input_shape[1]
        # initialize and define wavelet widths
        self.W_shape = self.nb_widths
        
        # self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        # self._trainable_weights = [self.W]
        
        init = tf.constant_initializer(2**np.arange(self.W_shape).astype('float32'))
        self.W = self.add_weight(
            shape=(self.W_shape,), initializer=init, trainable=True,
            name='{}_W'.format(self.name)
        )
        
        self.regularizers = []

        if self.kernel_regularizer:
            self.kernel_regularizer.set_param(self.W)
            self.regularizers.append(self.kernel_regularizer)

        if self.use_bias and self.bias_regularizer:
            self.bias_regularizer.set_param(self.b)
            self.regularizers.append(self.bias_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        # if self.kernel_constraint:
        #     self.constraints[self.W] = self.kernel_constraint
        # if self.use_bias and self.bias_constraint:
        #     self.constraints[self.b] = self.bias_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        # super(WaveletDeconvolution, self).build(input_shape)
        
    def call(self, x, mask=None):
        # shape of x is (batches, input_dim, input_len) if 'channels_first'
        # shape of x is (batches, input_len, input_dim) if 'channels_last'
        # we reshape x to channels first for computation
        if self.data_format == 'channels_last':
            x = tf.transpose(x, (0, 2, 1))

        #x = K.expand_dims(x, 2)  # add a dummy dimension for # rows in "image", now shape = (batches, input_dim, input_len, 1)
        
        # build the kernels to convolve each input signal with
        kernel_length = self.kernel_length
        T = (np.arange(0,kernel_length) - (kernel_length-1.0)/2).astype('float32')
        T2 = T**2
        # helper function to generate wavelet kernel for a given width
        # this generates the Mexican hat or Ricker wavelet. Can be replaced with other wavelet functions.
        '''
        Try some other wavelets kernels?
        https://pywavelets.readthedocs.io/en/latest/ref/cwt.html?highlight=kernel
        '''
        def gen_kernel(w):
            w2 = w**2
            B = (3 * w)**0.5
            A = (2.0 / (B * (np.pi**0.25)))
            mod = (1.0 - (T2)/(w2))
            gauss = K.exp(-(T2) / (2.0 * (w2)))
            kern = A * mod * gauss
            kern = K.reshape(kern, (kernel_length, 1))
            return kern
        
        # Morlet
        # def gen_kernel(w):
        #     w2 = w**2
        #     gauss = K.exp(-(T2) / (2.0 * (w2)))
        #     mod = tf.math.cos(5*T/w)
            
        #     kern = mod * gauss
        #     kern = K.reshape(kern, (kernel_length, 1))
        #     return kern
        
        
        
        wav_kernels = []
        for i in range(self.nb_widths):
            kernel = gen_kernel(self.W[i])
            wav_kernels.append(kernel)
        wav_kernels = tf.stack(wav_kernels, axis=0)
        # kernel, _ = tf.map_fn(fn=gen_kernel, elems=self.W)
        wav_kernels = K.expand_dims(wav_kernels, 0)
        wav_kernels = tf.transpose(wav_kernels,(0, 2, 3, 1))               

        # reshape input so number of dimensions is first (before batch dim)
        x = tf.transpose(x, (1, 0, 2))
        def gen_conv(x_slice):
            x_slice = K.expand_dims(x_slice,1) # shape (num_batches, 1, input_length)
            x_slice = K.expand_dims(x_slice,2) # shape (num_batches, 1, 1, input_length)
            return K.conv2d(x_slice, wav_kernels, strides=self.subsample, padding=self.padding, data_format='channels_first')
        outputs = []
        for i in range(self.input_dim):
            output = gen_conv(x[i,:,:])
            outputs.append(output)
        outputs = tf.stack(outputs, axis=0)
        # output, _ = tf.map_fn(fn=gen_conv, elems=x)
        outputs = K.squeeze(outputs, 3)
        outputs = tf.transpose(outputs, (1, 0, 3, 2))
        if self.data_format == 'channels_last':
            outputs = tf.transpose(outputs,(0, 2, 3, 1))
        return outputs
                
    def compute_output_shape(self, input_shape):
        out_length = conv_utils.conv_output_length(input_shape[2], 
                                                   self.kernel_length, 
                                                   self.padding, 
                                                   self.strides)        
        return (input_shape[0], self.input_dim, out_length, self.nb_widths)
    
    def get_config(self):
        config = {'nb_widths': self.nb_widths,
                  'kernel_length': self.kernel_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'kernel_constraint': self.kernel_constraint.get_config() if self.kernel_constraint else None,
                  'bias_constraint': self.bias_constraint.get_config() if self.bias_constraint else None,
                  'use_bias': self.use_bias}
        base_config = super(WaveletDeconvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   



"""
## Helper functions
We create two helper functions to help us get a sequence of
patches from the image, merge patches, and apply dropout.
"""


def window_partition(x, window_size):
    _, height, width, channels = x.shape

    patch_num_y = height // window_size[0]
    patch_num_x = width // window_size[1]
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size[0], patch_num_x, window_size[1], channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size[0], window_size[1], channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size[0]
    patch_num_x = width // window_size[1]
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size[0], window_size[1], channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output


"""
## Window based multi-head self-attention
Usually Transformers perform global self-attention, where the relationships between
a token and all other tokens are computed. The global computation leads to quadratic
complexity with respect to the number of tokens. Here, as the [original paper](https://arxiv.org/abs/2103.14030)
suggests, we compute self-attention within local windows, in a non-overlapping manner.
Global self-attention leads to quadratic computational complexity in the number of patches,
whereas window-based self-attention leads to linear complexity and is easily scalable.
"""


class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k
        # print('#'*40)
        # print(attn.shape)
        # print('#'*40)

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

'''
Modified Window Attention layer with external attention
part of the codes are modified from
https://keras.io/examples/vision/eanet/
'''
class WindowAttention_Ext(layers.Layer):
    def __init__(self, dim, window_size, num_heads, dim_coefficient=4,
                 attention_dropout =0, projection_dropout = 0, **kwargs):
        super(WindowAttention_Ext, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        # print('#'*40)
        # print('Window shape: {}'.format(self.window_size) )
        # print('#'*40)        
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        self.dim_coefficient = dim_coefficient
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        
        # two linear layers for M_k
        self.Mk_D1 = layers.Dense(dim * dim_coefficient)
        self.Mk_D2 = layers.Dense(dim // dim_coefficient)
        # two linear layers for M_v
        self.Mv_D1 = layers.Dense(dim // num_heads)
        self.Mv_D2 = layers.Dense(dim)
        
        self.att_dropout = layers.Dropout(attention_dropout)
        self.proj_dropout = layers.Dropout(projection_dropout)

    def build(self, input_shape):
        # pass
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

        # a layer for adjusting shape
        self.Ms = layers.Dense(input_shape[1], use_bias = False)
    def call(self, x, mask = None):
        _, num_patch, channels = x.shape
        
        # num_heads = self.num_heads * self.dim_coefficient
        # x = layers.Dense(self.dim * self.dim_coefficient)(x)
        x = self.Mk_D1(x) 
        # create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
        x = tf.reshape(
            x, shape=(-1, num_patch, self.num_heads * self.dim_coefficient, self.dim // self.num_heads)
        )
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # a linear layer M_k
        # attn = layers.Dense(self.dim // self.dim_coefficient)(x)
        # attn = self.Mk_D2(x)
        attn = self.Ms(x)

        # print('#'*40)
        # print('Attention shape: {}'.format(attn.shape) )
        # print('#'*40)
        # normalize attention map
        # attn = layers.Softmax(axis=2)(attn)
        # attn = keras.activations.softmax(attn, axis=2)

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        relative_position_bias= tf.concat([relative_position_bias for _ in range(self.dim_coefficient)], axis=0)
        relative_position_bias = tf.reshape(relative_position_bias, (-1, attn.shape[1],num_patch,num_patch))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
        
        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, num_patch, num_patch))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, num_patch, num_patch))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        # attn = layers.Dropout(self.attention_dropout)(attn)

        # dobule-normalization
        attn = attn / (1e-9 + tf.reduce_sum(attn, axis=-1, keepdims=True))
        attn = self.att_dropout(attn)

        # a linear layer M_v
        # x = layers.Dense(self.dim * self.dim_coefficient // num_heads)(attn)
        x = self.Mv_D1(attn)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [-1, num_patch, self.dim * self.dim_coefficient])
        # a linear layer to project original dim
        # x = layers.Dense(self.dim)(x)
        x = self.Mv_D2(x)
        # x = layers.Dropout(self.projection_dropout)(x)
        x = self.proj_dropout(x)
        
        return x
"""
## The complete Swin Transformer model
Finally, we put together the complete Swin Transformer by replacing the standard multi-head
attention (MHA) with shifted windows attention. As suggested in the
original paper, we create a model comprising of a shifted window-based MHA
layer, followed by a 2-layer MLP with GELU nonlinearity in between, applying
`LayerNormalization` before each MSA layer and each MLP, and a residual
connection after each of these layers.
Notice that we only create a simple MLP with 2 Dense and
2 Dropout layers. Often you will see models using ResNet-50 as the MLP which is
quite standard in the literature. However in this paper the authors use a
2-layer MLP with GELU nonlinearity in between.
"""


class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=[7,7],
        shift_size=[0,0],
        num_mlp=1024,
        att_type = 'MHSA',
        qkv_bias=True,
        dim_coefficient = 4,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes
        self.att_type = att_type

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        if self.att_type == 'MHSA':
            self.attn = WindowAttention(
                dim,
                window_size=self.window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )
        elif self.att_type == 'MHEA':
            self.attn = WindowAttention_Ext(
                dim,
                window_size=self.window_size,
                num_heads=num_heads,
                dim_coefficient=dim_coefficient,
                attention_dropout =dropout_rate,
                projection_dropout = dropout_rate
            )
        else:
            raiseExceptions('Attention type not yet implemented!')

        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )
        
        # y direction
        # print('#'*40)
        # print(self.num_patch)
        # print('#'*40)
        if self.num_patch[0] < self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.num_patch[0]
        
        # x direction
        if self.num_patch[1] < self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.num_patch[1]

    def build(self, input_shape):
        if self.shift_size == (0,0):
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size[0] * self.window_size[1]]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if min(self.shift_size) > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size[0] * self.window_size[1], channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size[0], self.window_size[1], channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if min(self.shift_size) > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


"""
## Model training and evaluation
### Extract and embed patches
We first create 3 layers to help us extract, embed and merge patches from the
images on top of which we will later use the Swin Transformer class we built.
"""


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1] * patches.shape[2]
        return tf.reshape(patches, (batch_size, patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim, direction = 'both'):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)
        self.direction = direction

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        if self.direction == 'both':
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = tf.concat((x0, x1, x2, x3), axis=-1)
            x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        
        elif self.direction =='x': # width
            x0 = x[:, :, 0::2, :]
            x1 = x[:, :, 1::2, :]
            x = tf.concat((x0, x1), axis=-1)
            x = tf.reshape(x, shape=(-1, height*(width // 2), 2 * C))

        elif self.direction =='y': #height
            x0 = x[:, 0::2, :, :]
            x1 = x[:, 1::2, :, :]
            x = tf.concat((x0, x1), axis=-1)
            x = tf.reshape(x, shape=(-1, (height // 2) * width, 2 * C))
        else:
            raiseExceptions('Must specify a valid direction for PatchMerging.')
        
        
        return self.linear_trans(x)

'''
CBAM and SE attention block
modified from
https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
'''
from tensorflow.keras.backend import image_data_format
def attach_attention_module(net, attention_module, ratio=8):
  if attention_module == 'se_block': # SE_block
    net = se_block(net, ratio=ratio)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net, ratio=ratio)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = layers.GlobalAveragePooling2D()(input_feature)
	se_feature = layers.Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = layers.Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	se_feature = layers.Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	if image_data_format() == 'channels_first':
		se_feature = layers.Permute((3, 1, 2))(se_feature)

	se_feature = layers.multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]
	
	shared_layer_one = layers.Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = layers.Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = layers.GlobalAveragePooling2D()(input_feature)    
	avg_pool = layers.Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = layers.GlobalMaxPooling2D()(input_feature)
	max_pool = layers.Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = layers.Add()([avg_pool,max_pool])
	cbam_feature = layers.Activation('sigmoid')(cbam_feature)
	
	if image_data_format() == "channels_first":
		cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)
	
	return layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = layers.Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = layers.Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if image_data_format() == "channels_first":
		cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)
		
	return layers.multiply([input_feature, cbam_feature])


##############################################################################
#   Different types of Correlation matrix 
################################################################################

'''
c by c learnable correlation/connectivity matrix
'''

class R_corr(layers.Layer):
    '''
    x: (B,T,C)
    a C by C symmetric matrix multiplied from the right
    '''
    def __init__(
        self, corr_passed = None, 
        init = tf.initializers.Zeros(),
        constraint = lambda x: tf.clip_by_value(x, -1.0, 1.0), **kwargs
    ):
        super(R_corr, self).__init__(**kwargs)
        self.corr_passed = corr_passed
        self.init = init
        self.constraint = constraint

    def build(self, input_shape):
        channels = input_shape[-1]
        # num_weights = (channels-1) * channels // 2

        # self.corr =  tf.Variable( shape = (channels, channels),
        #     initial_value=np.identity(channels), trainable=True
        # )

        # self.CM = (self.corr + tf.transpose(self.corr) )/2  # symmetry
        if self.corr_passed is None:
            self.corr = self.add_weight(
                            shape=(channels, channels),
                            initializer= self.init, constraint=self.constraint,
                            trainable=True,
                            name = 'r_corr'
                            )

            # self.corr = tf.Variable(
            #                 # shape=(channels, channels),
            #                 initial_value= np.zeros((channels, channels)),
            #                 trainable=True, dtype = tf.float32
            #                 )

            # self.CM = ( self.corr + tf.transpose(self.corr) )/2  # could be made into mean+var type
            self.CM = self.corr
        else:
            self.CM = tf.convert_to_tensor(self.corr_passed)
        


    def call(self, x):

        return x + tf.matmul(x, self.CM) + tf.matmul(x, tf.transpose(self.CM) )


'''
Custom layer for calculating
Kv where K(i,j) = exp(-d^2(xi, xj)/sigma) is a kernel function 
'''

class K_attention(layers.Layer):
    def __init__(self, **kwargs):
        super(K_attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        a T by T kernel matrix multiplied from the left
        '''
        # _,K_dim,_ = input_shape
        # self.r_sigma = tf.Variable(initial_value=0.0, trainable = True, dtype=tf.float32, name='r_sigma')
        self.r_sigma = self.add_weight(
                            shape=(1,),
                            initializer= tf.constant_initializer(value=0.01), 
                            constraint=lambda x: tf.clip_by_value(x, -0.01, 1e4),
                            trainable=True, name='r_sigma'
                            )
    def call(self, x, norm='euclidean', kernel='Gaussian', offdiag_mask=True):
        _,dim,f_dim = x.shape
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

        # K = K / tf.reduce_sum(K, axis= 1, keepdims=True)
        
        if offdiag_mask:
            offdiag_mask = 1.0 - tf.linalg.diag(tf.ones(dim, dtype=tf.float32))
            K = (offdiag_mask*K)

        return x + K @ x   #this looks like a correction form, adding learnable alpha infront?

'''
A slightly different version with learnable update step alpha
'''

class K_attention_alpha(layers.Layer):
    def __init__(self, **kwargs):
        super(K_attention_alpha, self).__init__(**kwargs)
    
    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        a T by T kernel matrix multiplied from the left
        '''
        # _,K_dim,_ = input_shape
        self.r_sigma = tf.Variable(initial_value=0.0, trainable = True, dtype=tf.float32, name='r_sigma')
        self.alpha = tf.Variable(initial_value=1.0, trainable = True, dtype=tf.float32, name='r_alpha')

    def call(self, x, norm='euclidean', kernel='Gaussian'):
        _,dim,f_dim = x.shape
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

        return x + self.alpha * K @ x   #this looks like a correction form, adding learnable alpha infront?


'''
a multiple heads version
'''
class K_attention_MH(layers.Layer):
    def __init__(self, num_heads=1, use_alpha = False, **kwargs):
        super(K_attention_MH, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.use_alpha = use_alpha
    
    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        a T by T kernel matrix multiplied from the left
        '''
        # _,K_dim,_ = input_shape
        # self.r_sigma = tf.Variable(initial_value=0.0, trainable = True, dtype=tf.float32, name='r_sigma')
        self.r_sigma = self.add_weight(
                            shape=(1,),
                            initializer= tf.constant_initializer(value=0.01), 
                            constraint=lambda x: tf.clip_by_value(x, -0.1, 2.0),
                            trainable=True, name='r_sigma'
                            )
        if self.use_alpha:
            self.alpha = tf.Variable(initial_value=1.0, trainable = True, dtype=tf.float32, name='r_alpha')

    def call(self, x, norm='euclidean', kernel='Gaussian', use_mask = False):
        _,size,channels = x.shape
        head_dim = channels//self.num_heads
        x = tf.reshape(x, shape=(-1,size, self.num_heads, head_dim))

        self_diff = tf.expand_dims(x, 2) - tf.expand_dims(x, 3)
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
        
        if self.use_alpha:
            y = x + self.alpha * K @ x  # other kinds of updates?
        else:
            y = x + K @ x

        y = tf.reshape(y, shape = (-1, size, channels))

        return  y


# %%
'''
qkv type with linear embedding
'''
class qKv_attention(layers.Layer):
    def __init__(self, dim, num_heads, dropout_rate, 
                 qkv_bias = True,**kwargs):
        super(qKv_attention, self).__init__(**kwargs)
        self.dim = dim 
        self.num_heads = num_heads
        self.qkv = layers.Dense(dim *3, activation = 'linear', 
                                use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        '''
        _,_,channels = input_shape
        self.r_sigma = tf.Variable(initial_value=0.0, trainable = True, dtype=tf.float32, name='r_sigma')
        self.proj = layers.Dense(channels, activation='linear', name='proj')

    def call(self, x, norm='euclidean', kernel='Gaussian', use_mask = False):     
        _, size, channels = x.shape

        x_qkv = self.qkv(x)
        head_dim = self.dim//self.num_heads
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        self_diff = tf.expand_dims(q, 2) - tf.expand_dims(k, 3)
        Dist = tf.norm(self_diff, ord = norm, axis=-1)
        Dist = tf.cast(Dist, dtype=tf.float32)
        # print('*'*40)
        # print(Dist.shape)
        # print('*'*40)
        if kernel == 'Gaussian':
            K = tf.math.exp(-Dist**2 * self.r_sigma) # need reduce mean here? scale adjustment?
        
        elif kernel == 'Linear':
            K = q @ tf.transpose(k, (0,1,3,2) )

        else:
            raiseExceptions('Kernel type not supported.')
        
        if use_mask:
            mask = np.zeros((size,size), dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask = tf.convert_to_tensor(mask)
            mask = tf.cast(mask[None,None, ...], tf.float32)
            K = K * mask

        x_qkv = K @ v
        # x_qkv = v + K @ v

        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, self.dim))

        
        x_proj = self.proj(x_qkv)

        return x_proj
        # return x + x_proj


'''
KAM with off-diagonal mask choice
'''

class K_attention_ex(layers.Layer):
    def __init__(self, use_margin=True, offdiag_mask = True, **kwargs):
        super(K_attention_ex, self).__init__(**kwargs)
        self.use_margin = use_margin
        self.offdiag_mask = offdiag_mask
    
    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        '''
        # _,K_dim,_ = input_shape
        # self.r_sigma = tf.Variable(initial_value=0.01, trainable = True, 
        #                            dtype=tf.float32, name='r_sigma')
        self.r_sigma = self.add_weight(
                                    shape=(1,),
                                    initializer= tf.constant_initializer(value=0.01), 
                                    constraint=lambda x: tf.clip_by_value(x, -0.05, 2.0),
                                    trainable=True, name='r_sigma'
                                    )
        if self.use_margin:
            self.margin = self.add_weight(
                                    shape=(1,),
                                    initializer= tf.constant_initializer(value=0.0), 
                                    # constraint=lambda x: tf.clip_by_value(x, 0, 2.0),
                                    trainable=True, name='margin'
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
            if self.use_margin:
                K = tf.math.exp(-Dist**2 * self.r_sigma+self.margin) # need reduce mean here? scale adjustment?
            else:
                K = tf.math.exp(-Dist**2 * self.r_sigma)
        
        else:
            raiseExceptions('Kernel type not supported.')
        
        if self.offdiag_mask: 
            offdiag_mask = 1.0 - tf.linalg.diag(tf.ones(size, dtype=tf.float32))
            K = (offdiag_mask*K)

        if use_mask:
            mask = np.zeros((size,size), dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask = tf.convert_to_tensor(mask)
            mask = tf.cast(mask[None, ...], tf.float32)
            K = K * mask

        return x + K @ x


'''
type of f(D, \theta) where f is manually specified
for mapping Distance matrix D to attention matrix (right)
'''
class Spatial_att(layers.Layer):
    def __init__(self, init, vmin, vmax, channel_wise=False, **kwargs):
        super(Spatial_att, self).__init__(**kwargs)
        self.init = init
        self.vmin = vmin
        self.vmax = vmax
        self.channel_wise = channel_wise

    def build(self, input_shape):
        _, _, fdim = input_shape
        if self.channel_wise:
            self.alpha= self.add_weight(
                                        shape=(fdim,),
                                        initializer= tf.constant_initializer(value=self.init), 
                                        constraint=lambda x: tf.clip_by_value(x, self.vmin, self.vmax),
                                        trainable=True, name='alpha'
                                        )
            self.margin = self.add_weight(
                                        shape=(fdim,),
                                        initializer= tf.constant_initializer(value=0.0), 
                                        # constraint=lambda x: tf.clip_by_value(x, 0, 2.0),
                                        trainable=True, name='margin'
                                        )
        else:
            self.alpha= self.add_weight(
                                        shape=(1,),
                                        initializer= tf.constant_initializer(value=self.init), 
                                        constraint=lambda x: tf.clip_by_value(x, self.vmin, self.vmax),
                                        trainable=True, name='alpha'
                                        )
            self.margin = self.add_weight(
                                        shape=(1,),
                                        initializer= tf.constant_initializer(value=0.0), 
                                        # constraint=lambda x: tf.clip_by_value(x, 0, 2.0),
                                        trainable=True, name='margin'
                                        )           
        
    
    def call(self, x, Ma, offdiag_mask=True):
        '''
        Ma: distance matrix passed
        '''
        if len(x.shape) == 3:
            _,size,f_dim = x.shape
        elif len(x.shape) == 4:
            _,_,size,f_dim = x.shape
        self.SaM = tf.math.exp(-self.alpha*Ma + self.margin)
        # self.SaM = tf.math.exp(-Ma*self.alpha + self.margin)
        # self.SaM = self.alpha*Ma

        if offdiag_mask:
            offdiag_mask = 1.0 - tf.linalg.diag(tf.ones(f_dim, dtype=tf.float32))
            self.SaM = self.SaM * offdiag_mask

        # normalization?
        self.SaM = self.SaM/(tf.reduce_sum(self.SaM, axis=1, keepdims=True)+1e-7)


        return x + x @ self.SaM 

'''
Predicting f using features' mutual distance
'''

class KAM_R(layers.Layer):
    def __init__(self, **kwargs):
        super(KAM_R, self).__init__(**kwargs)
        self.D1 = layers.TimeDistributed(layers.Dense(4, activation = 'elu'))
        self.D2 = layers.TimeDistributed(layers.Dense(4, activation = 'elu'))
        self.BN = layers.TimeDistributed(layers.BatchNormalization(axis=-1))
        self.Pred= layers.TimeDistributed(layers.Dense(1, activation = 'sigmoid'))
        # self.LN = layers.LayerNormalization()
    
    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        '''
        bsize,size,f_dim = input_shape
        self.reshape_1 = layers.Reshape([f_dim*f_dim, 1])
        self.reshape_2= layers.Reshape([f_dim,f_dim])
        # self.r_sigma = self.add_weight(
        #                             shape=(1,),
        #                             initializer= tf.constant_initializer(value=0.01), 
        #                             constraint=lambda x: tf.clip_by_value(x, -0.05, 2.0),
        #                             trainable=True, name='r_sigma'
        #                             )

    def call(self, x, norm='euclidean', offdiag_mask=True, use_mask = False):
        bsize,size,f_dim = x.shape
        
        # x = tf.transpose(x, (0,2,1))
        # self_diff = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
        # Dist = tf.norm(self_diff, ord = norm, axis=-1)
        # Dist = tf.cast(Dist, dtype=tf.float32)
        # # print('*'*40)
        # # print(Dist.shape)
        # # print('*'*40)
        # # Dist_tf_flatten = tf.reshape(Dist, [-1, size*size, 1])
        # # Dist_tf_flatten = layers.Reshape([f_dim*f_dim, 1])(Dist)
        # # norm_x = tf.reduce_sum(Dist, axis=1,keepdims=True) + 1e-7
        # Dist_tf_flatten = self.reshape_1(Dist)
        # # print('*'*40)
        # # print(Dist_tf_flatten.shape)
        # # print('*'*40)
        # Ma = self.D1(Dist_tf_flatten)

        # cor_x =  tf.transpose(x, (0,2,1)) @ x
        # norm_x = tf.reduce_sum(x**2, axis=1,keepdims=True) + 1e-7
        # c_flatten = self.reshape_1(cor_x/norm_x)       
        # Ma = self.D1(c_flatten)

        norm_x = x/(tf.reduce_sum(x**2, axis=1,keepdims=True) + 1e-7)**0.5
        cor_x = tf.transpose(norm_x, (0,2,1)) @ norm_x
        Ma = self.D1(cor_x[...,None])

        Ma1 = self.D2(Ma)
        Ma1 = self.BN(Ma1)
        Ma2 = self.Pred(Ma1)

        # self.Att = tf.reshape(Ma2, [-1, size, size])
        # self.Att = layers.Reshape([f_dim,f_dim])(Ma2)
        self.Att = self.reshape_2(Ma2)
        
        # normalization?
        self.Att = self.Att/(tf.reduce_sum(self.Att, axis=1, keepdims=True)+1e-7)
        # self.Att = self.LN(self.Att)
        
        if use_mask:
            mask = np.zeros((size,size), dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask = tf.convert_to_tensor(mask)
            mask = tf.cast(mask[None, ...], tf.float32)
            self.Att= self.Att * mask

        # self.Att = tf.math.exp(-self.r_sigma*self.Att)
        
        # x = tf.transpose(x, (0,2,1))

        if offdiag_mask:
            offdiag_mask = 1.0 - tf.linalg.diag(tf.ones(f_dim, dtype=tf.float32))
            return x + x @ (offdiag_mask*self.Att)
        
        else:
            return x + x @ self.Att

'''
Using a simple NN to approximate the mapping from distance matrix to attention matrix
with optional penalty specification
'''
class D2A(layers.Layer):
    def __init__(self, map_model, penalty_rate, num_kpts, **kwargs):
        super(D2A, self).__init__(**kwargs)
        self.map = map_model
        self.penalty_rate = tf.cast(penalty_rate, tf.float32)
        self.num_kpts = tf.cast(num_kpts, tf.int8)

            
    def build(self, input_shape):
        pass

    def call(self, x, DistM, offdiag_mask = True):
        if len(x.shape) == 3:
            _,size,f_dim = x.shape
        elif len(x.shape) == 4:
            _,_,size,f_dim = x.shape
        
        # DistM_tf = tf.convert_to_tensor(DistM, dtype=tf.float32) #tf1.x
        DistM_tf = tf.cast(DistM, dtype=tf.float32) #tf2.x

        DistM_tf_flatten = tf.reshape(DistM_tf, [-1])
        Ma = self.map(DistM_tf_flatten )

        self.Att = tf.reshape(Ma, [1, f_dim, f_dim])
        if offdiag_mask:
            offdiag_mask = 1.0 - tf.linalg.diag(tf.ones(f_dim, dtype=tf.float32))
            self.Att = self.Att * offdiag_mask
        # normalization?
        self.Att = self.Att/(tf.reduce_sum(self.Att, axis=1, keepdims=True)+1e-7)

        # check monotonic performance of the map
        # npts = tf.constant(21, dtype=tf.int32)
        key_points = tf.linspace(0.0, 1.0, self.num_kpts, name='kpoints')
        kpts = tf.reshape(key_points, [-1,1])
        kout = self.map(kpts)

        diff = kout[1:,0] - kout[:-1,0]
        penalty = 0.5*self.penalty_rate*tf.reduce_mean(tf.math.abs(diff) + diff)

        # self.add_loss(penalty)

        return x + x @ self.Att, penalty
        # return x @ self.Att

'''
adding constraints for controling the monocity when mapping feature correlation to attention
'''
class FC_mono(layers.Layer):
    def __init__(self, penalty_rate = 10, mono_mode=0,**kwargs):
        super(FC_mono, self).__init__(**kwargs)
        self.D1 = layers.Dense(4, activation = 'elu')
        #self.act = layers.Activation('elu')
        self.D2 = layers.Dense(4, activation = 'elu')
        self.BN = layers.BatchNormalization(axis=-1)
        self.Pred= layers.Dense(1, activation = 'sigmoid')
        # self.LN = layers.LayerNormalization()
        
        self.penalty_rate = penalty_rate
        self.mono_mode = mono_mode

    def build(self, input_shape):
        '''
        x must be of shape (B, T, C)
        '''
        # bsize,size,f_dim = input_shape
        # self.reshape_1 = layers.Reshape([f_dim*f_dim, 1])
        # self.reshape_2= layers.Reshape([f_dim,f_dim])
        # self.r_sigma = self.add_weight(
        #                             shape=(1,),
        #                             initializer= tf.constant_initializer(value=0.01), 
        #                             constraint=lambda x: tf.clip_by_value(x, -0.05, 2.0),
        #                             trainable=True, name='r_sigma'
        #                             )

    def call(self, x, norm='euclidean', offdiag_mask=True, use_mask = False):
        bsize,size,f_dim = x.shape
        
        # x = tf.transpose(x, (0,2,1))
        # self_diff = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
        # Dist = tf.norm(self_diff, ord = norm, axis=-1)
        # Dist = tf.cast(Dist, dtype=tf.float32)
        # # print('*'*40)
        # # print(Dist.shape)
        # # print('*'*40)
        # # Dist_tf_flatten = tf.reshape(Dist, [-1, size*size, 1])
        # # Dist_tf_flatten = layers.Reshape([f_dim*f_dim, 1])(Dist)
        # # norm_x = tf.reduce_sum(Dist, axis=1,keepdims=True) + 1e-7
        # Dist_tf_flatten = self.reshape_1(Dist)
        # # print('*'*40)
        # # print(Dist_tf_flatten.shape)
        # # print('*'*40)
        # Ma = self.D1(Dist_tf_flatten)

        norm_x = x/(tf.reduce_sum(x**2, axis=1,keepdims=True) + 1e-7)**0.5
        cor_x = tf.transpose(norm_x, (0,2,1)) @ norm_x
        # c_flatten = self.reshape_1(cor_x/norm_x)
        # c_input = cor_x[...,None]/norm_x[...,None]
        Ma = self.D1(cor_x[...,None])
        #Ma = self.act(Ma)
        #Ma = self.BN(Ma)
        Ma1 = self.D2(Ma)
        Ma1 = self.BN(Ma1)
        #Ma1 = self.act(Ma1)
        Ma2 = self.Pred(Ma1)

        # self.Att = tf.reshape(Ma2, [-1, size, size])
        # self.Att = layers.Reshape([f_dim,f_dim])(Ma2)
        # self.Att = self.reshape_2(Ma2)
        self.Att = Ma2[...,0]

        # normalization?
        self.Att = self.Att/(tf.reduce_sum(self.Att, axis=1, keepdims=True)+1e-7)
        # self.Att = self.LN(self.Att)

        # for constraining the monotoncity
        key_points = tf.linspace(-1.0, 1.0, 21, name='kpoints')
        kpts = tf.reshape(key_points, [-1,1,1,1])
        kout = self.D1(kpts)
        #kout = self.act(kout)
        #kout = self.BN(kout)
        kout = self.D2(kout)
        kout = self.BN(kout)
        #kout = self.act(kout)
        kout = self.Pred(kout)
        
        diff_L = (kout[1:11,0,0,0] - kout[:10,0,0,0])
        diff_R = (kout[11:,0,0,0] - kout[10:-1,0,0,0])

        if self.mono_mode == 'D':
            penalty = 0.5*self.penalty_rate*tf.reduce_mean(tf.math.abs(diff_L) + diff_L + tf.math.abs(diff_R) + diff_R)
        elif self.mono_mode == 'I':
            penalty = 0.5*self.penalty_rate*tf.reduce_mean(tf.math.abs(diff_L) - diff_L + tf.math.abs(diff_R) - diff_R)
        elif self.mono_mode == 'DI':
            penalty = 0.5*self.penalty_rate*tf.reduce_mean(tf.math.abs(diff_L) + diff_L + tf.math.abs(diff_R) - diff_R)
        elif self.mono_mode == 'ID':
            penalty = 0.5*self.penalty_rate*tf.reduce_mean(tf.math.abs(diff_L) - diff_L + tf.math.abs(diff_R) + diff_R)
        else:
            penalty = 0.0
            print('No penalty for monotocity')
       
        if use_mask:
            mask = np.zeros((size,size), dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask = tf.convert_to_tensor(mask)
            mask = tf.cast(mask[None, ...], tf.float32)
            self.Att= self.Att * mask

        # self.Att = tf.math.exp(-self.r_sigma*self.Att)
        
        # x = tf.transpose(x, (0,2,1))

        if offdiag_mask:
            offdiag_mask = 1.0 - tf.linalg.diag(tf.ones(f_dim, dtype=tf.float32))
            return x + x @ (offdiag_mask*self.Att), penalty
        
        else:
            return x + x @ self.Att, penalty
