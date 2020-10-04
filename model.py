import tensorflow as tf

from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, Input
from tensorflow.keras.layers import DepthwiseConv2D, Lambda, MaxPool2D

import numpy as np
import random 

def PEPX(x, nf1, nf2, nf3, nf4, name):
    # The PEPX-module is described as
    # conv1x1 -> conv1x1 -> DWConv3x3 -> conv1x1 -> conv1x1

    # Project input features to a lower dimension
    # Expand features to a higher dimension that is different than that of the input features 
    # Project features back to a lower dimension, and 
    # Extend channel dimensionality to a higher dimension to produce the ﬁnal features.

    x = Conv2D(filters=nf1, kernel_size=1, strides=1, activation="relu", name=name + "_Project1")(x)
    x = Conv2D(filters=nf2, kernel_size=1, strides=1, activation="relu", name=name + "_Expand")(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", name=name + "_DW")(x)
    x = Conv2D(filters=nf3, kernel_size=1, strides=1, activation="relu", name=name + "_Project2")(x)
    x = Conv2D(filters=nf4, kernel_size=1, strides=1, activation="relu", name=name + "_Extend")(x)
    return x

def covid_net(input_shape, nf, fc_units, single=False):
    """ 
    Replication of the COVID-Net architecture. 
    UL = upper layer in covid-net.
    nf: number of filters in the first PEPX module
    fc_units: number of units in the first fully connected layer
    """
    
    img = Input(shape=input_shape)

    pep_nf = nf

    # Input: 224x224x1, output: 112x112x64
    x = Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", padding="same", name="CV_7x7")(img)

    # UL 1. Input: 112x112x64, output: 56x56xnf
    y1 = Conv2D(filters=pep_nf, kernel_size=1, activation="relu", name="CV_UL1")(x)
    y1 = MaxPool2D(2, name="MP_UL1")(y1)

    # PEPX1.1 - PEPX1.3. 
    # 1.1       Input: 112x112x64, output: 56x56xnf
    # 1.2-1.3.  Input: 56x56xnf, output: 56x56xnf
    pepx1_1 = PEPX(x, 32, 48, 32, pep_nf, "P11")
    pepx1_1 = MaxPool2D(2, name="MP_P11")(pepx1_1)
    
    pepx1_2 = keras.layers.add([pepx1_1, y1])
    pepx1_2 = PEPX(pepx1_2, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P12")

    pepx1_3 = keras.layers.add([pepx1_1, pepx1_2, y1])
    pepx1_3 = PEPX(pepx1_3, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf,  "P13")

    # UL 2. Input: 56x56xnf, output: 28x28x2nf
    y2 = keras.layers.add([pepx1_1, pepx1_2, pepx1_3, y1])
    y2 = Conv2D(filters=2 * pep_nf, kernel_size=1, activation="relu", name="CV_UL2")(y2)
    y2 = MaxPool2D(2, name="MP_UL2")(y2)

    # PEPX 2.1-2.4 
    # 2.1       Input: 56x56xnf, output: 28x28x2nf
    # 2.2-2.4   Input: 28x28x2nf, output: 28x28x2nf
    pepx2_1 = keras.layers.add([pepx1_1, pepx1_2, pepx1_3, y1])
    pepx2_1 = PEPX(pepx2_1, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, 2 * pep_nf, "P21")
    pepx2_1 = MaxPool2D(2, name="MP_P21")(pepx2_1)

    pep_nf *= 2

    pepx2_2 = keras.layers.add([pepx2_1, y2])
    pepx2_2 = PEPX(pepx2_2, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P22")

    pepx2_3 = keras.layers.add([pepx2_1, pepx2_2, y2])
    pepx2_3 = PEPX(pepx2_3, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P23")

    pepx2_4 = keras.layers.add([pepx2_1, pepx2_2, pepx2_3, y2])
    pepx2_4 = PEPX(pepx2_4, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P24")

    # UL 3. Input: 28x28x2nf, output: 14x14x4nf
    y3 = keras.layers.add([pepx2_1, pepx2_2, pepx2_3, pepx2_4, y2])
    y3 = Conv2D(filters=2 * pep_nf, kernel_size=1, activation="relu", name="CV_UL3")(y3)
    y3 = MaxPool2D(2, name="MP_UL3")(y3)

    # PEPX 3.1-3.6. 
    # 3.1       Input: 28x28x2nf, output: 14x14x4nf
    # 3.2-3.6   Input: 14x14x4nf, output: 14x14x4nf
    pepx3_1 = keras.layers.add([pepx2_1, pepx2_2, pepx2_3, pepx2_4, y2])
    pepx3_1 = PEPX(pepx3_1, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, 2 * pep_nf, "P31")
    pepx3_1 = MaxPool2D(2, name="MP_P31")(pepx3_1)

    pep_nf *= 2

    pepx3_2 = keras.layers.add([pepx3_1, y3])
    pepx3_2 = PEPX(pepx3_2, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P32")

    pepx3_3 = keras.layers.add([pepx3_1, pepx3_2, y3])
    pepx3_3 = PEPX(pepx3_3, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P33")

    pepx3_4 = keras.layers.add([pepx3_1, pepx3_2, pepx3_3, y3])
    pepx3_4 = PEPX(pepx3_4, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P34")

    pepx3_5 = keras.layers.add([pepx3_1, pepx3_2, pepx3_3, pepx3_4, y3])
    pepx3_5 = PEPX(pepx3_5, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P35")

    pepx3_6 = keras.layers.add([pepx3_1, pepx3_2, pepx3_3, pepx3_4, pepx3_5, y3])
    pepx3_6 = PEPX(pepx3_6, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P36")

    # UL 4. Input: 14x14x4nf, output: 7x7x8nf
    y4 = keras.layers.add([pepx3_1, pepx3_2, pepx3_3, pepx3_4, pepx3_5, pepx3_6, y3])
    y4 = Conv2D(filters=2 * pep_nf, kernel_size=1, activation="relu", name="CV_UL4")(y4)
    y4 = MaxPool2D(2, name="MP_UL4")(y4)

    # PEPX 4.1-4.3
    # 4.1       Input: 14x14x4nf, output: 7x7x8nf
    # 4.2-4.3   Input: 7x7x8nf, output: 7x7x8nf
    pepx4_1 = keras.layers.add([pepx3_1, pepx3_2, pepx3_3, pepx3_4, pepx3_5, pepx3_6, y3])
    pepx4_1 = PEPX(pepx4_1, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, 2 * pep_nf, "P41")
    pepx4_1 = MaxPool2D(2, name="MP_P41")(pepx4_1)

    pep_nf *= 2

    pepx4_2 = keras.layers.add([pepx4_1, y4])
    pepx4_2 = PEPX(pepx4_2, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P42")

    pepx4_3 = keras.layers.add([pepx4_1, pepx4_2, y4])
    pepx4_3 = PEPX(pepx4_3, pep_nf // 2, 3 * pep_nf // 4, pep_nf // 2, pep_nf, "P43")

    # Flatten and two fully connected layers, possibly dropout
    z = keras.layers.add([pepx4_1, pepx4_2, pepx4_3, y4], name="OUTPUTX")
    
    flat = Flatten(name="Flat")(z)   # 100352x1
    flat = keras.layers.Dropout(rate=0.50)(flat)
    
    fc1 = Dense(units=fc_units, activation="relu", name="FC1")(flat)
    fc1 = keras.layers.Dropout(rate=0.20)(fc1)
    
    fc2 = Dense(units=256, activation="relu", name="FC2")(fc1)
    fc2 = keras.layers.Dropout(rate=0.20)(fc2)
    
    if single:
        output = Dense(units=3, name="Output_before_SM")(fc2)
        output = Activation("softmax", name="Output_after_SM")(output)
        return Model(img, output)
        
    return Model(img, fc2)

def siamese_net(nf, fc_units):
    input_shape = (224, 224, 1)
    
    covid_model = covid_net(input_shape, nf, fc_units)
    
    img1 = Input(shape=input_shape, name="Img1")  # First image
    img2 = Input(shape=input_shape, name="Img2")  # Second image
    
    feature_vec1 = covid_model(img1)
    feature_vec2 = covid_model(img2)

    l1 = Lambda(lambda tensors : keras.backend.abs(tensors[0] - tensors[1]), name="Lambda")([feature_vec1, feature_vec2])
    similarity = Dense(units=1, activation="sigmoid", name= "Similarity")(l1)
    
    model = Model(inputs=[img1, img2], outputs=similarity) 
    return model
    
