import numpy as np 
import pandas as pd
import sys
import h5py
import matplotlib.pyplot as plt 

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Activation,BatchNormalization
from keras.optimizers import rmsprop,adagrad,Adam

#from sklearn.model_selection import train_test_split

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

num_classes = 10
df3 = pd.read_csv(train_file,header=None,delimiter=" ",dtype=np.uint8)
df3_test = pd.read_csv(test_file,header=None,delimiter=" ",dtype=np.uint8)

#saving to hd5 files
target = df3[3072]

train = df3.drop([3072],axis=1).values
test = df3_test.drop([3072],axis=1).values
target = to_categorical(target,10)
def shape_image(data):
    data = np.moveaxis(data.reshape(data.shape[0],-1,32,32),1,-1)
    return data
trainx = train[:40000,:]
valx = train[40000:,:]
trainy = target[:40000,:]
valy = target[40000:,:]
trainx = shape_image(trainx)
valx = shape_image(valx)
testx = shape_image(test)
#train_df = np.loadtxt(train_file,delimiter=" ",dtype=np.uint8)
print("train data loaded ...")

# h5f = h5py.File("data_preprocessed.h5",'w')
# h5f.create_dataset('trainx',data=trainx)
# h5f.create_dataset("trainy",data=trainy)
# h5f.create_dataset("valx",data=valx)
# h5f.create_dataset("valy",data=valy)
# h5f.close()
trainx = trainx/255
valx = valx/255
testx = testx/255

# test_df = np.loadtxt(test_file,delimiter=" ",dtype=np.uint8)
# testx = test_df[:,:-1]
# testx = np.moveaxis(testx.reshape((testx.shape[0],-1,32,32)),1,-1)
# testx = testx/255

"""
Architecture of network
#+++++++++++++++++++++++++++++++++#
"""
def create_network():
            model = Sequential()
            #conv layer 1
            model.add(Conv2D(64,kernel_size=3,input_shape=(32,32,3)))
            model.add(Activation('relu'))

            #maxpooling layer 1
            model.add(MaxPooling2D(pool_size=(2,2),strides=1,padding="same"))

            #conv layer 2
            model.add(Conv2D(128,kernel_size=3))
            model.add(Activation("relu"))

            #maxpool layer 2
            model.add(MaxPooling2D(pool_size=(2,2),strides=1,padding="same"))

            #flatten the nodes from 2d images to one d
            model.add(Flatten())

            #dense layer 1
            model.add(Dense(512))
            model.add(Activation("relu"))

            #dense layer 2
            model.add(Dense(256))
            model.add(Activation("relu"))

            #output layer
            model.add(Dense(num_classes))
            model.add(BatchNormalization())
            model.add(Activation("softmax"))
            return model


def train_network(model,optimizer,metrics='accuracy',loss='categorical_crossentropy',batch_size=100,nepochs=10):
            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=[metrics])

            model.fit(trainx,trainy,
                      batch_size=batch_size,
                      epochs=nepochs,
                      shuffle=True,
                      validation_data=(valx,valy),
                      verbose=1)
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
'''
#+++++++++++++++++++++++++++++++++++++++++++++#
'''

model1 = create_network()
print("model architecture created...")

#optimizers

#train model
print("training model ...")
epochs = 1
lr = 0.001
batch_size=100
opt = 'adam'
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999,epsilon=1e-8)
model1 = train_network(model1,optimizer,nepochs=epochs,batch_size=batch_size)

print("model optimized")
predictions = model1.predict(testx)
predictions = np.argmax(predictions,axis=1)

training_loss,val_loss,training_acc,val_acc = get_hist(model1)
fig,ax = plt.subplots(1,2,figsize = (10,5))
axes=ax.ravel()
plot_func(training_loss,"loss for adam","epochs","loss","training_loss",axes[0])
plot_func(val_loss,"loss for adam","epochs","loss","val_loss",axes[0])
plot_func(training_acc,"accuracy for adam","epochs","accuracy","training_acc",axes[1])
plot_func(val_acc,"accuracy for adam","epochs","accuracy","val_acc",axes[1])
plt.savefig(opt+"_"+str(lr)+"_"+"_b1_0.9_metrics_accuracy.png")

with open(output_file,"w") as f:
    f.write("\n".join(x for x in predictions))




