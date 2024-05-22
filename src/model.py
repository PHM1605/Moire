import os
import numpy as np

from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, Multiply, Maximum

def createModel(height, width, depth):
#     num_epochs = 20 # 50 26 200 # we iterate 200 times over the entire training set
    kernel_size_1 = 7 # we will use 7x7 kernels 
    kernel_size_2 = 3 # we will use 3x3 kernels 
    pool_size = 2 # we will use 2x2 pooling throughout
    conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 16 # ...switching to 16 after the first pooling layer
    drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size = 32 # 128 512 the FC layer will have 512 neurons

    inp_row = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
    inp_col = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)

    conv_1_row = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inp_row)
    conv_1_col = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inp_col)

    pool_1_row = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_row)
    pool_1_col = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_col)

    inp_merged = Multiply()([pool_1_row, pool_1_col])
    C4 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(inp_merged)
    S2 = MaxPooling2D(pool_size=(4, 4))(C4)
    drop_1 = Dropout(drop_prob_1)(S2)
    C5 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(drop_1)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(S3)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = Dropout(drop_prob_1)(S4)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(2, activation='softmax')(drop_3)
    
    model = Model(inputs=[inp_row, inp_col], outputs=out) # To define a model, just specify its input and output layers
    
    return model

if __name__ == "__main__":
    model = createModel(1920, 2560, 3)
    input = np.zeros((12, 1920, 2560, 3))
    output = model([input, input])
    print(output.shape)