#Your code goes here
#%%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import math
import sys

# class neuron:
#     def __init__(self,number_inputs):
#         self.number_inputs = number_inputs

def sigmoid_func(X):
        return 1/(1+np.exp(-X))

def softmax(y):
    y_pred_max = -np.max(y,axis=1,keepdims=True)
    y = np.exp(y+y_pred_max)
    column_wise_sum = np.sum(y,axis=1,keepdims=True)
    return (y/column_wise_sum)

def relu_func(X):
        return np.max((0,X))
    
def tanh_func(X):
        return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def sigmoid_prime(X):
    return sigmoid_func(X)*(1-sigmoid_func(X))

def softmax_prime(y):
    return softmax(y)*(1-softmax(y))

#%%
class neuron_layer:
    def __init__(self,inputs_neurons,number_of_neurons):
        self.inputs_neurons=inputs_neurons
        self.number_of_neurons = number_of_neurons
        self.set_weights(np.zeros((inputs_neurons,number_of_neurons)))
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
            y_pred_max = -np.max(self.layer_output_z,axis=1,keepdims=True)
            layer_output_z_dummy = np.exp(self.layer_output_z + y_pred_max)
            column_sum = np.sum(layer_output_z_dummy,axis=1,keepdims=True)
            self.activate_a = layer_output_z_dummy/column_sum
    



    


#%%
class Neural_a:

    def __init__(self,trainX,trainY,n_output_neurons,n_hl_neurons,ismulti=False):
        assert type(n_hl_neurons)==list
        # assert (len(trainY.shape))==2
        self.trainX = trainX
        self.trainY = trainY
        self.n_layers = len(n_hl_neurons)
        self.n_output_neurons = n_output_neurons
        self.ismulti = ismulti
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

        self.layers = [neuron_layer(self.trainX.shape[1],neurons_list[0])]
        self.layers =self.layers + [neuron_layer(neurons_list[i-1],neurons_list[i]) for i in range(1,len(neurons_list))]
        self.layers = self.layers+ [neuron_layer(neurons_list[-1],self.n_output_neurons)]

    def forward_propagate(self,x):
        self.layers[0].layer_output(x)
        # self.z_outputs = [np.hstack((np.ones((self.trainX.shape[0],1)),self.layers[0].layer_output_z))]
        for i in range(1,len(self.layers)):
            self.layers[i-1].activate_layer(sigmoid=True)
            self.layers[i].layer_output(self.layers[i-1].activate_a)
            # self.z_outputs += [np.hstack((np.ones((self.trainX.shape[0],1)),self.layers[i].layer_output_z))]  #inputs trainX data
        if(self.ismulti==True):
            self.layers[-1].activate_layer(softmax=True)
        else:
            self.layers[-1].activate_layer(sigmoid=True)
        return self.layers[-1].layer_output_z

    def Binary_cross_entropy_loss(self,y,outputs):
        term1 = y*np.log(outputs[:,0])
        term2 = (1-y)*np.log(1-outputs[:,0])
        return -(1/y.shape[0])*np.sum(term1+term2)
    
    def multiclass_ce_loss(self,y,outputs):
        return -1*(np.sum(y*np.log(outputs))/y.shape[0])
    
    def multiclass_ce_derivative(self,y):
        assert self.layers[-1].activate_a.shape ==y.shape
        return self.layers[-1].activate_a - y


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
        
        
    
    def back_propagate(self,x,y,ismulti=False):

        #initializing all weights to zeros
        back_prop_weights_updates = [np.zeros((w.get_weights().shape)) for w in self.layers]
        back_prop_bias_updates = [np.zeros(b.get_bias().shape) for b in self.layers]

        #feed forward 
        self.forward_propagate(x)

        #delta for output layer
        if(ismulti==True):
            delta = self.multiclass_ce_derivative(y)
        else:
            delta = self.cost_derivative(y)  #dl/dz

        #dw and db for output layer
        back_prop_weights_updates[-1] = np.matmul(self.layers[-2].activate_a.T,delta)
        back_prop_bias_updates[-1] = np.sum(delta,axis=0,keepdims=True)

        for i in range(1,len(self.layers)):
            curr_layer = self.layers[-(i+1)]
            curr_z = curr_layer.layer_output_z
            sp = self.layers[-(i+1)].activate_a*(1-self.layers[-(i+1)].activate_a)
            #doubt since calculation formula is not correct i guess
            delta = np.matmul(delta,self.layers[-i].get_weights().T)*sp
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
                    new_weights = self.layers[j].get_weights() - (learning_rate/(np.sqrt(i)))*w
                    self.layers[j].set_weights(new_weights)
                for j,b in enumerate(db):
                    new_biases = self.layers[j].get_bias() - (learning_rate/(np.sqrt(i)))*b
                    self.layers[j].set_bias(new_biases)
            # ou = sigmoid_func(self.forward_propagate(self.trainX))
            # print(self.Binary_cross_entropy_loss(self.trainY,ou))
        return self.layers[-1].activate_a

        



#%%
if __name__=="__main__":

    input_file = sys.argv[1]
    param_file = sys.argv[2]
    weights_file = sys.argv[3]
    # input_file = "train.csv"
    # param_file = "param.txt"
    # weights_file = "weight.txt"

    param_read =  open(param_file,'r')
    params=[]
    for line in param_read:
        params.append(line.strip())
        
    df = pd.read_csv(input_file,header=None)
    last_column = len(df.columns)-1
    trainy = df[last_column].values.reshape((df.shape[0],1))
    trainx  = df.drop([last_column],axis=1).values
    m,n = trainx.shape
    # trainx=np.hstack((np.ones((m,1)),trainx))
    if(params[0]=='1'):
        learning_rate = float(params[1])
        iterations = int(params[2])
        batch_size = int(params[3])
        neural_net_arct =params[4].split(",")
        neural_net_arct = [int(i) for i in neural_net_arct]
        n_of_hidden_layers = len(neural_net_arct)

        neural_net = Neural_a(trainx,trainy,1,neural_net_arct)
        _ = neural_net.train_network(learning_rate,iterations,batch_size)
        weights = [np.vstack((w.get_bias(),w.get_weights())) for w in neural_net.layers]
    if(params[0]=='2'):
        learning_rate=float(params[1])
        iterations = int(params[2])
        batch_size = int(params[3])
        neural_net_arct =params[4].split(",") 
        neural_net_arct = [int(i) for i in neural_net_arct]        
        n_of_hidden_layers = len(neural_net_arct)

        neural_net = Neural_a(trainx,trainy,1,neural_net_arct)
        _ = neural_net.train_network(learning_rate,iterations,batch_size,True)
        weights = [np.vstack((w.get_bias(),w.get_weights())) for w in neural_net.layers]
    outputs = sigmoid_func(neural_net.forward_propagate(trainx))
    loss = neural_net.Binary_cross_entropy_loss(trainy,outputs)
    print(outputs)
    print(loss)
    # outputs = [1 if x>0.7 else 0 for x in outputs]
    # accuracy = [1 if int(x)==int(y[0]) else 0 for (x,y) in zip(outputs,trainy.tolist())]
    with open(weights_file,"w") as f:
        
        for wt in weights:
                wt = wt.flatten().tolist()
                for w in wt:
                    f.write(str(w))
                    f.write("\n")






#%%

