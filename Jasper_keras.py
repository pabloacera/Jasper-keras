#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:49:04 2020

@author: labuser
"""

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, \
BatchNormalization, GlobalAveragePooling1D, Softmax, Add, Dense, Activation


def sub_block(x, n_filter, kernel_s, droppout):
    '''
    Sub-block of jasper
    '''
    x = Conv1D(filters=n_filter, 
               kernel_size=kernel_s,
               padding='same',
              )(x)
    x = _bn_relu(x)
    x = Dropout(rate=droppout)(x)
    return x


def last_sub_block(x, n_filter, kernel_s, droppout, prev_input):
    '''
    Last sub-block contain a residual connection with
    the previous block
    '''
    
    x = Conv1D(filters=n_filter, 
               kernel_size=kernel_s,
               padding='same',
               )(x)
    x = BatchNormalization()(x)
    '''
    #  1X1 conv residual connection
    '''
    x1 = Conv1D(filters=n_filter, 
                kernel_size=1, 
                padding='same', 
                )(prev_input[-1])
    x1 = BatchNormalization()(x1)
    '''
    #  1X1 conv residual connection
    '''
    if len(prev_input)==1:
        x = Add()([x1,x])
        x = Activation("relu")(x)
        x = Dropout(rate=droppout)(x)
    else:
        x = Add()(prev_input[:-1] + [x1,x])
        x = Activation("relu")(x)
        x = Dropout(rate=droppout)(x)
    
    return x


def keras_jasper(inputs, R, B, output, Deep=None):
    '''
    Jasper BxR model has B blocks, each with R subblocks. 
    Each sub-block applies the following operations: 
        1Dconvolution, 
        batch norm, 
        ReLU, 
        dropout.
    All sub-blocks in a block have the same number of output channels.
    Each block input is connected directly into the last subblock via 
    a residual connection. The residual connection is first projected 
    through a 1x1 convolution to account for different numbers of input 
    and output channels, then through a batch norm layer. The output
    of this batch norm layer is added to the output of the batch norm
    layer in the last sub-block. The result of this sum is passed 
    through the activation function and dropout to produce the output 
    of the current block.
    All Jasper models have four additional convolutional
    blocks: one pre-processing and three post-processing.
    https://arxiv.org/pdf/1904.03288.pdf
    '''
    x = inputs
    
    # pre-processing (prolog) Conv layer
    x1 = Conv1D(filters=48, 
               kernel_size=3,
               padding='same',
               dilation_rate=2
               )(x)
    x1 = _bn_relu(x1)
    
    ## First block
    
    n_filter = 48
    kernel_s = 2
    droppout = 0.15
    
    block_track = [x1]
    
    for i in range(R):
        
        for j in range(B-1):
    
            if i == 0 and j == 0:
                    
                x = sub_block(x1, 
                              n_filter, 
                              kernel_s,
                              droppout)

            else:
                 x = sub_block(x, 
                               n_filter, 
                               kernel_s,
                               droppout)
        
        x = last_sub_block(x, 
                           n_filter, 
                           kernel_s,
                           droppout,
                           block_track)
        
        block_track.append(x)
   
    # epilog conv layer
    x = Conv1D(filters=48, 
               kernel_size=3,
               dilation_rate=2,
               padding='same')(x)
    x =_bn_relu(x)
    
    x = Conv1D(filters=48, 
               kernel_size=3,
               dilation_rate=2,
               padding='same')(x)
    x =_bn_relu(x)
    
    x = Conv1D(filters=1, 
               kernel_size=1, 
               padding='same', 
               activation='relu')(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(output, activation='sigmoid')(x)
    
    return x

inputs = Input(shape=(100, 2))

output = keras_jasper(inputs, 4, 3, 1, Deep=True)
output = build_Jasper(inputs, 1)

model = Model(inputs=inputs, outputs=output)

model.summary()

