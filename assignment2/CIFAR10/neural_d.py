#Your code goes here
#%%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import math
import sys
from sklearn.preprocessing import StandardScaler
import scipy.fftpack as fp
from skimage.feature import hog
# from mnist import MNIST


#one hot encoder 
def fft_image_data(trainx):
    #shape of trainx = m,n
    ## Functions to go from image to frequency-image and back
    im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                                   axis=1)
    freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                                 axis=0)

    ## Read in data file and transform
    for i in range(trainx.shape[0]):
        curr_img = trainx[i,:].reshape((32,32))
        data = curr_img

        freq = im2freq(data)
        back = freq2im(freq)
        assert(np.allclose(data, back))

        ## Helper functions to rescale a frequency-image to [0, 255] and save
        remmax = lambda x: x/x.max()
        remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
        touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)
        trainx[i,:] = touint8(freq).reshape((1,1024))
    return trainx

def hog_features(trainx):
    #shape of trainx = m,n
    ## Functions to go from image to frequency-image and back

    ## Read in data file and transform
    hog_features=[]
    hog_images =[]
    for i in range(trainx.shape[0]):
        curr_img = trainx[i,:].reshape((32,32))
        data = curr_img
        fd, hog_image = hog(curr_img, orientations=8, pixels_per_cell=(2,2),
                    cells_per_block=(2, 2), visualize=True)
        hog_features.append(fd)
        hog_images.append(hog_image)
        ## Helper functions to rescale a frequency-image to [0, 255] and save
    hog_feat_arr = np.array(hog_features)
    hog_images_arr = np.array(hog_images)
    return hog_feat_arr,hog_images_arr

def onehotEncoder(array,k_class):
    if(type(array)==list):
        array = np.array(array)
    assert len(array.shape)==1
    onehotencoded = np.zeros((array.shape[0],k_class))
    array = array.reshape((array.shape[0],))
    for i in range(array.shape[0]):
        onehotencoded[i,array[i]-1] = 1
    return onehotencoded

def sigmoid_func(X):
        return 1/(1+np.exp(-X))

def softmax_stable(y):
    y_pred_max = -1*np.max(y,axis=1,keepdims=True)
    y = np.exp(y+y_pred_max)
    column_wise_sum = np.sum(y,axis=1,keepdims=True)
    return (y/column_wise_sum)

def relu_func(X):
        X = np.where(X<0,0,X)
        return X

def relu_derivative(x):
    x = np.where(x<0,0,1)
    return x
    
def tanh_func(X):
        return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def sigmoid_prime(X):
    return sigmoid_func(X)*(1-sigmoid_func(X))

def softmax_prime(y):
    return softmax_stable(y)*(1-softmax_stable(y))

class neuron_layer:
    def __init__(self,inputs_neurons,number_of_neurons):
        self.inputs_neurons=inputs_neurons
        self.number_of_neurons = number_of_neurons
        self.set_weights(np.sqrt(2/(inputs_neurons+number_of_neurons))*np.random.uniform(-1,1,(inputs_neurons,number_of_neurons)))
        self.set_bias(np.zeros((1,number_of_neurons)))

    def set_weights(self,weights):
        self.weights = weights
    
    def get_weights(self):
        return self.weights
    
    def set_bias(self,bias):
        self.bias = bias
    
    def get_bias(self):
        return self.bias

    def layer_output(self,inputs):
        self.layer_output_z = np.matmul(inputs,self.weights) +self.bias
    
    def activate_layer(self,sigmoid=False,relu=False,tanh=False,softmax=False):
        self.activate_a=None
        if(sigmoid==True):
            self.activate_a = sigmoid_func(self.layer_output_z)
            # if(isoutput==False):
            #     self.activate_a = np.hstack((np.ones((self.activate_a.shape[0],1)),self.activate_a))
        if(relu==True):
            self.activate_a = relu_func(self.layer_output_z)
            # if(isoutput==False):
            #     self.activate_a = np.hstack((np.ones((self.activate_a.shape[0],1)),self.activate_a))
        if(tanh==True):
            self.activate_a = tanh_func(self.layer_output_z)
            # if(isoutput==False):
            #     self.activate_a = np.hstack((np.ones((self.activate_a.shape[0],1)),self.activate_a))
        if(softmax==True):
            self.activate_a = softmax_stable(self.layer_output_z)
    



    


class Neural_a:

    def __init__(self,trainX,trainY,n_output_neurons,n_hl_neurons,ismulti=False,activation_func="sigmoid"):
        assert type(n_hl_neurons)==list
        # assert (len(trainY.shape))==2
        self.trainX = trainX
        self.trainY = trainY

        #activation function for hidden layer, wheather sigmoid or relu or tanh
        self.activation_func = activation_func

        #numbe of hidden layers
        self.n_layers = len(n_hl_neurons)

        #number of units in output layer
        self.n_output_neurons = n_output_neurons

        #whether classification is multiclass or binary
        self.ismulti = ismulti

        #to create architecture of network
        self.create_architechure(n_hl_neurons)
    
    def create_architechure(self,neurons_list):
        if(self.ismulti==True):
            assert self.n_output_neurons > 1
        '''
        neurons list is a list containing number of neurons in each hidden layer
        length of list should be equal to number of hidden layers
        it should not have input layer and output layer shape
        '''
        assert type(neurons_list)==list

        #first hidden layer
        self.layers = [neuron_layer(self.trainX.shape[1],neurons_list[0])]

        #adding hidden layer upto output layer
        self.layers =self.layers + [neuron_layer(neurons_list[i-1],neurons_list[i]) for i in range(1,len(neurons_list))]

        #output layer which is last layer
        self.layers = self.layers+ [neuron_layer(neurons_list[-1],self.n_output_neurons)]

    def forward_propagate(self,x):

        #to feed input to first hidden layer
        self.layers[0].layer_output(x)
        for i in range(1,len(self.layers)):
            #activation of hidden layer

            if(self.activation_func =="sigmoid"):
                self.layers[i-1].activate_layer(sigmoid=True)
            if(self.activation_func =='relu'):
                self.layers[i-1].activate_layer(relu=True)
            
            #feeding next hidden layer
            self.layers[i].layer_output(self.layers[i-1].activate_a)

        #activation of last layer 
        if(self.ismulti==True):
            self.layers[-1].activate_layer(softmax=True)
        else:
            self.layers[-1].activate_layer(sigmoid=True)
        return self.layers[-1].activate_a

    def Binary_cross_entropy_loss(self,y,outputs):
        term1 = y*np.log(outputs[:,0])
        term2 = (1-y)*np.log(1-outputs[:,0])
        return -(1/y.shape[0])*np.sum(term1+term2)
    
    def multiclass_ce_loss(self,y,outputs):
        return -1*(np.sum(y*np.log(outputs+1e-6))/y.shape[0])
    
    def multiclass_ce_derivative(self,y):
        assert self.layers[-1].activate_a.shape ==y.shape
        return (self.layers[-1].activate_a - y)/y.shape[0]


    def cost_derivative(self,trainy):
        #shape of y --> m*2
        #shape of t --> m*1
        y  = self.layers[-1].activate_a 
        t = trainy
        return (y-t)/trainy.shape[0]   #dl/dy
    

    
    def mini_batch(self,batch_size):
        number_of_mini_batch=math.floor(self.trainX.shape[0]/batch_size)
        self.batches_x = []
        self.batches_y =[]
        for i in range(number_of_mini_batch):
            self.batches_x.append(self.trainX[i*batch_size:(i+1)*batch_size,:])
            self.batches_y.append(self.trainY[i*batch_size:(i+1)*batch_size,:])
        if(number_of_mini_batch*batch_size<self.trainX.shape[0]):
            self.batches_x.append(self.trainX[number_of_mini_batch*batch_size:,:])
            self.batches_y.append(self.trainY[number_of_mini_batch*batch_size:,:])
        
        return self.batches_x,self.batches_y
        
    def accuracy(self,y,outputs):
        '''
        y and outputs are of size m,k where k is number of classes
        '''
        y =np.argmax(y,axis=1)
        outputs = np.argmax(outputs,axis=1)
        acc = [1 if x==y else 0 for (x,y) in zip(y,outputs)]
        return float(np.sum(acc)/len(acc))
    
    def predictions(self,x):
        pred_outputs = self.forward_propagate(x)
        return pred_outputs

    
    def back_propagate(self,x,y):

        #initializing all weights to zeros
        back_prop_weights_updates = [np.zeros((w.get_weights().shape)) for w in self.layers]
        back_prop_bias_updates = [np.zeros(b.get_bias().shape) for b in self.layers]

        #feed forward 
        self.forward_propagate(x)

        #delta for output layer
        if(self.ismulti==True):
            delta = self.multiclass_ce_derivative(y)
        else:
            delta = self.cost_derivative(y)  #dl/dz

        #dw and db for output layer
        back_prop_weights_updates[-1] = np.matmul(self.layers[-2].activate_a.T,delta)
        back_prop_bias_updates[-1] = np.sum(delta,axis=0,keepdims=True)

        for i in range(1,len(self.layers)):
            curr_layer = self.layers[-(i+1)]
            prev_layer = self.layers[-i]
            if(self.activation_func=='sigmoid'):
                sp = curr_layer.activate_a*(1-curr_layer.activate_a)
            if(self.activation_func=='relu'):
                sp = relu_derivative(curr_layer.layer_output_z)
            #doubt since calculation formula is not correct i guess
            delta = np.matmul(delta,prev_layer.get_weights().T)*sp
            if(i==(len(self.layers)-1)):
                activations = x.T
            else:
                activations = self.layers[-(i+2)].activate_a.T
            # print(back_prop_weights_updates[-(i+1)].shape)
            back_prop_weights_updates[-(i+1)]= np.dot(activations,delta)
            back_prop_bias_updates[-(i+1)]= np.sum(delta,axis=0,keepdims=True)
        
        return back_prop_weights_updates,back_prop_bias_updates

    def train_network(self,learning_rate,iterations,n_batches,adaptive=False):
        self.mini_batch(n_batches)
        x,y = self.batches_x,self.batches_y
        k = len(x)
        cost_history=[]
        for i in range(iterations):
            # weights = [np.zeros(w.get_weights().shape) for w in self.layers]
            # biases =  [np.zeros(w.get_bias().shape) for w in self.layers]
            mini_batchx,mini_batchy = x[i%k],y[i%k]
            dw,db = self.back_propagate(mini_batchx,mini_batchy)
            # weights = [w+w_a for w,w_a in zip(weights,dw)]
            # biases = [b+b_a for b,b_a in zip(biases,db)]
            # print(self.Binary_cross_entropy_loss())
            if(adaptive==False):
                for j,w in enumerate(dw):
                    new_weights = self.layers[j].get_weights() - (learning_rate)*w
                    self.layers[j].set_weights(new_weights)
                for j,b in enumerate(db):
                    new_biases = self.layers[j].get_bias() - (learning_rate)*b
                    self.layers[j].set_bias(new_biases)
            
            if(adaptive==True):
                for j,w in enumerate(dw):
                    new_weights = self.layers[j].get_weights() - (learning_rate/(np.sqrt(i+1)))*w
                    self.layers[j].set_weights(new_weights)
                for j,b in enumerate(db):
                    new_biases = self.layers[j].get_bias() - (learning_rate/(np.sqrt(i+1)))*b
                    self.layers[j].set_bias(new_biases)
            
            outputs = self.forward_propagate(mini_batchx)
            cost =self.multiclass_ce_loss(mini_batchy,outputs)
            cost_history.append(cost)
            
            if(i%100==0):
                print("iteration no.: "+ str(i)+"   cost:"+str(cost))
            # ou = sigmoid_func(self.forward_propagate(self.trainX))
            # print(self.Binary_cross_entropy_loss(self.trainY,ou))
        return self.layers[-1].activate_a,cost_history

        



#%%
if __name__=="__main__":
    # mdata = MNIST("samples")
    # images, labels = mdata.load_training()
    # images = np.array(images)
    # labels = np.array(labels).reshape((images.shape[0],))

    scaler = StandardScaler()
    input_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

        
    df = pd.read_csv(input_file,header=None,error_bad_lines = False)
    df_test = pd.read_csv(test_file,header=None,error_bad_lines = False)    
    last_column = len(df.columns)-1
    last_column_test = len(df_test.columns)-1
    trainy = df[last_column].values.reshape((df.shape[0],))
    trainy=trainy+1
    trainy = onehotEncoder(trainy,10)
    trainx  = df.drop([last_column],axis=1).values

    trainx,trainx_hog= hog_features(trainx)
    trainx = scaler.fit_transform(trainx)

    testx = df_test.drop([last_column_test],axis=1).values
    testx = hog_features(testx)
    # testx= testx/255.0
    testx = scaler.fit_transform(testx)

    neural_net = Neural_a(trainx,trainy,10,[517],ismulti=True,activation_func='relu')
    _,cost_hist = neural_net.train_network(0.001,2000,100)
    weights = [np.vstack((w.get_bias(),w.get_weights())) for w in neural_net.layers]
    outputs = neural_net.forward_propagate(testx)
    outputs = np.argmax(outputs,axis=1).tolist()
    output_train = neural_net.forward_propagate(trainx)
    accu = neural_net.accuracy(output_train,trainy)
    plt.plot(cost_hist)
    plt.show()
    print(accu)
    # weights_file = weights_file[:-4]+str(iterations)+weights_file[-4:]
    with open(output_file,"w") as f:
        
        for o in outputs:
                    f.write(str(o))
                    f.write("\n")

#%%
