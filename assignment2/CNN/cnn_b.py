import numpy as np
np.random.seed(1337)
import keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,rmsprop

import sys
import pandas as pd
import matplotlib.pyplot as plt 

def preprocess_inputs(data):
    data = np.moveaxis(data.reshape(data.shape[0],-1,32,32),1,-1)
    data = data.astype(np.float32)
    data[...,0]-= data[...,0].mean()
    data[...,1]-= data[...,1].mean()
    data[...,2]-= data[...,2].mean()
    return data

def conv_block(input, nfilters,dropout=None,bottleneck=False, weight_decay=1e-4):
    x = BatchNormalization()(input)
    x = Activation("relu")(x)

    if(bottleneck):
        n_filters = 4*nfilters
        x= Conv2D(n_filters,1,padding="same",use_bias=False,kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    x = Conv2D(nfilters,3,kernel_initializer='he_normal',use_bias=False,padding='same')(x)
    if(dropout):
        x = Dropout(dropout)(x)
    
    return x

def dense_block(x,nb_layers,nFilters,filterNumGrowth,bottleneck=False,dropout=None,growFilters=True):
    feature_map = [x]
    for i in range(nb_layers):
        fm = conv_block(x,filterNumGrowth,dropout,bottleneck)
        feature_map.append(fm)
        x = concatenate([x,fm],axis=-1)
        if(growFilters):
              nFilters = nFilters+filterNumGrowth
    return feature_map,x,nFilters

def transition_block(x,nfilters,compression=0.5):
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    nfilters = int(nfilters*compression)
    x = Conv2D(nfilters,1,kernel_initializer='he_normal',padding='same')(x)
    x = AveragePooling2D((2,2),strides=(2,2))(x)
    return x


'''
#nb_layers_per_dense_block  ---> number of layers in denseblock which should be (depth-4)%3==0
nb_filters = initial_number of filters to provide
'''
def DenseNet(input,nb_layers_per_dense_block=6,nb_filter=12,numDenseBlock=3,dropout_rate=None,bottleneck=False,growth_rate=12,initial_sampling=False):
    initial_numFilters = 2*growth_rate
    nb_filter= initial_numFilters
    x = Conv2D(nb_filter,3,strides=1,kernel_initializer='he_normal',use_bias=False,padding='same')(input)
    if(initial_sampling):
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3),strides=2,padding='same')(x)

    features_per_dense_block=[]
    for i in range(numDenseBlock-1):
        fm_,x,nb_filter = dense_block(x,nb_layers_per_dense_block,nb_filter,12,bottleneck=bottleneck,dropout=dropout_rate)
        features_per_dense_block.append(fm_)
        x = transition_block(x,nb_filter)
        nb_filter = 0.5*nb_filter

    fm_,x,nb_filter = dense_block(x,nb_layers_per_dense_block,nb_filter,12,bottleneck=bottleneck,dropout=dropout_rate)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10)(x)
    x=Activation("softmax")(x)
    return x

def create_model(input_shape):
    img_input=Input(shape=input_shape)
    x = DenseNet(img_input,bottleneck=True)
    model = Model(img_input,x,name='DenseNet')
    return model

def get_hist(model):
  training_loss = model.history.history['loss']
  val_loss = model.history.history['val_loss']
  training_acc = model.history.history['acc']
  val_acc = model.history.history['val_acc']
  return training_loss,val_loss,training_acc,val_acc

def plot_func(data,title,xlabel,ylabel,legend,ax):
    ax.plot(data,label=legend)
    ax.legend(loc='upper center',shadow=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

if __name__=='__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    num_classes = 10
    print("loading files ...")
    train_df = pd.read_csv(train_file,delimiter=" ",dtype=np.uint8,header=None)
    test_df = pd.read_csv(test_file,header=None,delimiter=" ",dtype=np.uint8)
    print("Files loaded")

    target = train_df[3072]
    target = to_categorical(target, num_classes)

    train = train_df.drop([3072],axis=1).values
    testx = test_df.drop([3072],axis=1).values

    def shape_data(data):
        return np.moveaxis(data.reshape((data.shape[0],-1,32,32)),1,-1)

    print("reshapping data files...")
    train = preprocess_inputs(train)
    testx = preprocess_inputs(testx)

    print("data reshaped")

    trainx = train[:45000,:,:,:]
    valx = train[45000:,:,:,:]
    trainy = target[:45000,:]
    valy = target[45000:,:]

    print("creating model ....")
    model = create_model((32,32,3))
    print("compiling model ...")
    optimizer = Adam(lr = 0.01)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    print("training model ...")
    
    generator = ImageDataGenerator(rotation_range=15,width_shift_range=5./32,height_shift_range=5./32,horizontal_flip=True)
    generator.fit(trainx,seed=0)

    model.fit_generator(generator.flow(trainx,trainy,batch_size=300),validation_data=(valx,valy),verbose=2,epochs=35)
    print('model trained ...')

    predictions = model.predict(testx)
    predictions = np.argmax(predictions,axis=1)

    training_loss,val_loss,training_acc,val_acc = get_hist(model)
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    axes=ax.ravel()
    plot_func(training_loss,"loss for adam","epochs","loss","training_loss",axes[0])
    plot_func(val_loss,"loss for adam","epochs","loss","val_loss",axes[0])
    plot_func(training_acc,"accuracy for adam","epochs","accuracy","training_acc",axes[1])
    plot_func(val_acc,"accuracy for adam","epochs","accuracy","val_acc",axes[1])
    plt.savefig("partb_default_rmsprop_epochs15_metrics_accuracy.png")

    with open(output_file,"w") as f:
        f.write("\n".join(str(x) for x in predictions))











