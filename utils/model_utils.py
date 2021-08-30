import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input
from keras_unet_collection.layer_utils import *

import sys
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/keras-unet-collection/')

def dummy_loader(model_path):
    print('Import model:\n{}'.format(model_path))
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def Euclidean_distance(x, y):
    """Computes the Euclidean_distance between two tensorflow variables
    """
    d = math_ops.reduce_mean(math_ops.square(math_ops.subtract(x, y)), axis=-1)
    return d

def triplet(y_true, y_pred, N=128, margin=5.0):
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]
    
    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)
    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

def gan_loss(y_true, y_pred, N=128, margin=5.0):
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]
    
    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)
    loss = tf.maximum(0., margin - d_pos + d_neg)
    loss = tf.reduce_mean(loss)
    return loss


def denoise_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                       stack_num_down=2, stack_num_up=1, activation='ReLU', 
                       batch_norm=False, pool=True, unpool=True, name='unet3plus'):
    
    depth_ = len(filter_num_down)

    X_encoder = []
    X_decoder = []

    X = input_tensor
    
    X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down, 
                   activation=activation, batch_norm=batch_norm, name='{}_down_0'.format(name))
    
    X_encoder.append(X)

    # downsampling levels
    for i, f in enumerate(filter_num_down[1:]):

        X = encode_layer(X, f, 2, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_{}'.format(name, i+1))

        X = CONV_stack(X, f, kernel_size=3, stack_num=stack_num_down, 
                       activation=activation, batch_norm=batch_norm, name='{}_en_conv_{}'.format(name, i+1))

        X_encoder.append(X)

    # treat the last encoded tensor as the first decoded tensor
    X_decoder.append(X_encoder[-1])

    # upsampling levels
    X_encoder = X_encoder[::-1]

    # loop over upsampling levels
    for i, f in enumerate(filter_num_skip):

        # collecting tensors for layer fusion
        X_fscale = []

        # for each upsampling level, loop over all available downsampling levels (similar to the unet++)
        for lev in range(depth_):

            # counting scale difference between the current down- and upsampling levels
            pool_scale = lev-i-1 # -1 for python indexing

            # one scale deeper input is obtained from the nearest **decoder** output
            if pool_scale == -1:

                X = decode_layer(X_decoder[i], f, 2, unpool, kernel_size=3, activation=None, 
                                 batch_norm=True, name='{}_up_{}_de_{}'.format(name, i, i))

            # other inputs are obtained from **encoder** outputs
            else:
                # deeper tensors are upsampled
                if pool_scale < 0:
                    pool_size = 2**(-1*pool_scale)

                    X = decode_layer(X_encoder[lev], f, pool_size, unpool, kernel_size=3, activation=None, 
                                     batch_norm=True, name='{}_up_{}_en_{}'.format(name, i, lev))

                # unet skip connection (identity mapping)    
                elif pool_scale == 0:

                    X = X_encoder[lev]

                # shallower tensors are downsampled
                else:
                    pool_size = 2**(pool_scale)

                    X = encode_layer(X_encoder[lev], f, pool_size, pool, activation=activation, 
                                     batch_norm=batch_norm, name='{}_down_{}_en_{}'.format(name, i, lev))

            # a conv layer after feature map scale change
            X = CONV_stack(X, f, kernel_size=3, stack_num=1, activation=activation, 
                           batch_norm=batch_norm, name='{}_fsdown_from_{}_to_{}'.format(name, i, lev))

            X_fscale.append(X)  

        # layer fusion at the end of each level
        # stacked conv layers after concat. BatchNormalization is fixed to True

        X = concatenate(X_fscale, axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, filter_num_aggregate, kernel_size=3, stack_num=stack_num_up, 
                       activation=activation, batch_norm=True, name='{}_fusion_conv_{}'.format(name, i))
        X_decoder.append(X)

        # return decoder outputs
    return X_decoder


def denoise_sup_head(X_decoder, filter_num_sup, activation='ReLU', 
                     batch_norm=False, pool=True, unpool=True, name='unet3plus'):
    OUT_stack = []
    X_decoder = X_decoder[::-1]
    L_out = len(X_decoder)

    for i in range(1, L_out-1):   
        pool_size = 2**(i)

        X = CONV_stack(X_decoder[i], filter_num_sup, kernel_size=3, stack_num=1, activation=activation, 
                       batch_norm=batch_norm, name='{}_output_conv0_{}'.format(name, i))

        X = decode_layer(X, filter_num_sup, pool_size, unpool, kernel_size=3, activation=None, 
                         batch_norm=True, name='{}_output_sup{}'.format(name, i))

        X = Conv2D(1, 3, padding='same', name='{}_output_conv1_{}'.format(name, i))(X)

        OUT_stack.append(X)

    X = CONV_stack(X_decoder[0], filter_num_sup, kernel_size=3, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_output_conv0_final'.format(name))

    X = Conv2D(1, 3, padding='same', name='{}_output_conv1_final'.format(name))(X)

    OUT_stack.append(X)
    
    return OUT_stack
    
    
