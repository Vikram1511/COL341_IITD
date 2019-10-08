#%%
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd 
import math
from pandas.api.types import is_string_dtype
from pandas.plotting import table
from pandas.api.types import is_numeric_dtype
from sklearn.utils import class_weight

import sys
import tqdm
from tqdm import tqdm
import csv
import time


def sigmoid(x):
    return 1/(1+np.exp(-x))

def hypothesis(theta,x):
    assert x.shape[1]==theta.shape[0]
    return np.matmul(x,theta)

def cost(theta,X,Y):
    m,n  = X.shape
    return -(1/(m)*(np.sum(Y*np.log(hypothesis(theta,X))+(1-Y)*np.log(1-hypothesis(theta,X)))))

def softmax_func(theta,X,class_weights=None):
    Y_pred = hypothesis(theta,X)
    y_pred_max = -np.max(Y_pred,axis=1).reshape((Y_pred.shape[0],1))
    # Y_pred = Y_pred - y_pred_max
    Y_pred  = np.exp(Y_pred + y_pred_max)
    sum_row_wise = np.sum(Y_pred,axis=1)
    sum_row_wise = sum_row_wise.reshape((sum_row_wise.shape[0],1))
    softmax_prob = Y_pred / sum_row_wise #shape --> m,k
    if(class_weights is None==False):
        softmax_prob = softmax_prob*class_weights
    return softmax_prob 

def log_likelihood(theta,X,Y_true):
    m = Y_true.shape[0]
    softmax_cross_entropy = softmax_func(theta,X)
    log_likelihood = -ma.log(np.multiply(softmax_cross_entropy,Y_true)).filled(0)
    logL = float(np.sum(log_likelihood)/(2*m))
    return logL



def onehotEncoder(array,k_class):

    '''
    @array -  encoded array
    @k_class - number of unique classes in array
    '''
    #to ensure the type of input is one dimensional array
    if(type(array)==list):
        array = np.array(array)
    assert len(array.shape)==1

    #initializing (m,k_classes) array as one hot encoded array
    onehotencoded = np.zeros((array.shape[0],k_class))
    array = array.reshape((array.shape[0],))

    #loop for each data point in array
    for i in range(array.shape[0]):
        onehotencoded[i,array[i]-1] = 1
    return onehotencoded

def decode(array,mapping):

    '''
        @mapping- python dict where keys and values and encoded and true values respectively
        @array - encoded array
    '''
    #convert to list type
    array_list = array.tolist()
    for i in range(array.shape[0]):
        array_list[i] = mapping[array[i]-1]
    return np.array(array_list)

def encoder(array,unique_classes):

    '''
        @array -  input categorical feature which is to be encoded 
        @unique_classes -  input list of unique classes in array
    '''
    array = array.reshape((array.shape[0],))

    #initializing a ditionary to store mapping of true classes to encoded value
    mapping={}
    for i in range(len(unique_classes)):
        mapping[unique_classes[i]]=i+1

    #encoding
    for j in range(array.shape[0]):
        if(array[j] not in list(mapping.keys())):
            array[j] = None
        else:
            array[j] = list(mapping.keys()).index(array[j])+1
    #returning encoded array
    return array.reshape((array.shape[0],))

def read_inputs(trainFile,testFile):
    df_train = pd.read_csv(trainFile,header=None)
    df_test = pd.read_csv(testFile,header=None)
    no_of_columns = len(df_train.columns)
    Y = df_train[no_of_columns-1].values.reshape((df_train.shape[0],1))
    df_train.drop(no_of_columns-1,axis=1,inplace=True)
    label_order = [['usual', 'pretentious', 'great_pret'], 
        ['proper', 'less_proper', 'improper', 'critical', 'very_crit'], 
        ['complete', 'completed', 'incomplete', 'foster'], 
        ['1', '2', '3', 'more'], 
        ['convenient', 'less_conv', 'critical'], 
        ['convenient', 'inconv'], 
        ['nonprob', 'slightly_prob', 'problematic'], 
        ['recommended', 'priority', 'not_recom'],
        ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']]

    for i,column in enumerate(df_train.columns):
        if(is_string_dtype(df_train[column])):
            uniquec = label_order[i]
            df_train[column] = pd.Series(encoder(df_train[column].values,uniquec))
            if(column in df_test.columns):
                df_test[column] = pd.Series(encoder(df_test[column].values,uniquec))  
                
    #print(Y.shape)
    Y_encoded = encoder(Y,label_order[-1])
    X = df_train.values
    X = np.hstack((np.ones((X.shape[0],1)),X))
    X_test = df_test.values 
    X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))
    return X,Y_encoded,X_test,label_order[-1]

def mini_batches(X,y,batch_size):
    m,n = X.shape
    n_batches = math.floor(m/batch_size)
    miniB = []
    for i in range(n_batches):
        mini_batch_x = X[i*batch_size:(i+1)*batch_size,:]
        mini_batch_y = y[i*batch_size:(i+1)*batch_size,:]
        miniB.append((mini_batch_x,mini_batch_y))
    if(m > n_batches*batch_size):
        mini_batch_x = X[n_batches*batch_size:,:]
        mini_batch_y = y[n_batches*batch_size:,:]
        miniB.append((mini_batch_x,mini_batch_y))
    return miniB

def gradients(theta,X,y,class_weights=None,lambdaa=None):
    #term2 = 0
    gradient = np.zeros(theta.shape)  # shape (n,k)
    m,n = X.shape
    softmax_entropy = softmax_func(theta,X,class_weights)   # shape (m,k)
    gradient =  (-np.matmul(X.T,y) + np.matmul(X.T,softmax_entropy))/m
    # for i in range(gradient.shape[1]):
    #     bool_mask = np.where(y[:,i].reshape((m,))==1)
    #     masked = X[bool_mask,:][0]
    #     xi_sum = np.sum(masked,axis=0)       # shape 1*n
    #     xi_prob_multiply =  X*softmax_entropy[:,i].reshape((m,1))     #shape m*n
    #     xi_prob_multiply = np.sum(xi_prob_multiply,axis=0).reshape((1,n))          #shape 1*n
    #     gradient[:,i] = 1/m*(-xi_sum + xi_prob_multiply).ravel()
    #gradient= np.matmul(X.T,y) - np.matmul(X.T,softmax_func(theta,X))
    if(lambdaa):
        pass
    return gradient



def run_model(X,Y,learning_rate,iteration,adaptive=False,alpha=None,Beta=None,batch_size=None,class_weights=None):
    m,n = X.shape
    k = Y.shape[1]
    theta = np.zeros((n,k))
    if(batch_size):
        mini_batch = mini_batches(X,Y,batch_size)
    else:
        mini_batch = [[X,Y]]
    cost_history = []
    for i in range(iteration):
        mini_cost = 0
        for batch in mini_batch:
            x_mini = batch[0]
            y_mini = batch[1]
            cost =log_likelihood(theta,x_mini,y_mini)
            gradients_ = gradients(theta,x_mini,y_mini,class_weights)
            #print(np.dot(gradients_.ravel(),gradients_.ravel()))

            ## if adaptive learning rate
            if(adaptive==True):
                theta = theta - (learning_rate/np.sqrt(i+1))*gradients_

            # if alpha-Beta line tracking
            elif(alpha!=None and Beta!=None):
                theta_ad = theta - learning_rate*gradients_
                a = log_likelihood(theta_ad,x_mini,y_mini)
                b =  cost - alpha*learning_rate*np.dot(theta.ravel(),theta.ravel())
                while(log_likelihood(theta_ad,x_mini,y_mini)>cost - alpha*learning_rate*np.dot(gradients_.ravel(),gradients_.ravel())):
                    learning_rate=Beta*learning_rate
                    theta_ad = theta - learning_rate*gradients_
                theta = theta - learning_rate*gradients_

            # fixed learning rate
            else:
                theta = theta - learning_rate*gradients_
            mini_cost = mini_cost+cost
        cost_history.append(mini_cost)
        if(i>2):
                mini_cost_prev =cost_history[-2]
                if(mini_cost_prev-mini_cost < (mini_cost_prev*(0.001/100))):
                    break
        i=i+1
    return cost_history,theta,gradients_

def prediction(theta,X,mapping):
   prediction_softmax = softmax_func(theta,X)
   predictions = np.argmax(prediction_softmax,axis=1) +1
   prediction_map = []
   for i in range(predictions.shape[0]):
       ind=predictions[i]-1
       prediction_map.append(mapping[ind])
   return prediction_map,predictions.reshape((X.shape[0],1))

def kfolds(X,Y,k):
    n = X.shape[0]
    m_k = round(n/k)
    data = np.hstack((X,Y))
    K_folds = []

    for i in range(k):
        if i < k-1:
            K_folds.append(data[m_k*i:m_k*(i+1),:])
        else :
            K_folds.append(data[m_k*i:,:])
    return K_folds

def cross_validation(X,Y,K,lr,iteration,adaptive=False,lasso=False,alpha=None,Beta=None,batch_size=None,lamda=None):
    m,n = X.shape 
    m,k = Y.shape 
    Kfolds  = kfolds(X,Y,K)
    cross_val = []
    max_micro = 0
    for i in range(K):
        #vertical stacking of other than validation set of kfolds array
        train_set = Kfolds[:i]+Kfolds[i+1:]
        train_set = np.vstack((train_set))

        #trainX and trainY distinguishing
        trainX = train_set[:,:train_set.shape[1]-k]
        trainY = train_set[:,train_set.shape[1]-k:]

        #validation set 
        val_set = Kfolds[i]

        #valx and valy distinguishing
        valX = val_set[:,:val_set.shape[1]-k]
        valY = val_set[:,val_set.shape[1]-k:]
        
        #optimum weights from ridge normal equation
        if(lasso==True):
            pass
        if(alpha!=None and Beta!=None):
            _,theta,_ = run_model(trainX,trainY,lr,iteration,alpha=alpha,Beta=Beta)
        if(adaptive==True):
            _,theta,_ = run_model(trainX,trainY,lr,iteration)


        #calculating loss value over validation set
        pred = np.argmax(softmax_func(theta,valX),axis=1)+1
        pred = onehotEncoder(pred,5)
        microf1,_,_ = micro_scores(valY,pred)
        if(microf1>max_micro):
            max_micro = microf1

        #appending this loss value in cross_val array
        cross_val.append(microf1)
    print(max_micro)
    return cross_val

def confusion_matrix(y_true,y_pred,onehotencoded=True):
    if(onehotencoded==True):
        m,k = y_true.shape
        return np.matmul(y_true.T,y_pred).astype(int)

def normalized_confusion_marix(y_true,y_pred):
    confmatrix = confusion_matrix(y_true,y_pred)
    confmatrix = (confmatrix/np.sum(confmatrix,axis=1).reshape((confmatrix.shape[0],1))).astype(float)
    return confmatrix

def precesion_recall(y_true,y_pred):
    #precesion = TP/TP+FP
    #recall = TP/TP+FN
    confmatrix  = confusion_matrix(y_true,y_pred)
    row_sum_confmatrix = np.sum(confmatrix,axis=0).tolist()
    column_sum_confmatrix= np.sum(confmatrix,axis=1).tolist()
    precesion = []
    recall = []
    tp = []
    for i in range(confmatrix.shape[0]):
        for j in range(confmatrix.shape[1]):
            if(i==j):
                tp.append(confmatrix[i][j])
                precesion.append(round(float(confmatrix[i][j]/row_sum_confmatrix[i]),4))
                recall.append(round(float(confmatrix[i][j]/column_sum_confmatrix[i]),4))
    fp = np.subtract(np.array(row_sum_confmatrix), np.array(tp)).tolist()
    fn =  np.subtract(np.array(column_sum_confmatrix), np.array(tp)).tolist()
    return precesion,recall,tp,fp,fn


def micro_scores(y_true,y_pred):
    # avg_prec = float(np.nansum(precesion)/len(precesion))
    # avg_rec = float(np.nansum(recall)/len(recall))
    # f1score = 2*(avg_prec*avg_rec)/(avg_prec+avg_rec)
    _,_,tp,fp,fn = precesion_recall(y_true,y_pred)
    tp_sum = sum(tp)
    tp_fp_sum = sum(tp)+sum(fp)
    tp_fn_sum = tp_sum + sum(fn)
    micro_avg_prec = round(float(tp_sum/tp_fp_sum),3)
    micro_avg_rec = round(float(tp_sum/tp_fn_sum),3)
    micro_avg_f1 = round((2*micro_avg_prec*micro_avg_rec)/(micro_avg_prec+micro_avg_rec),3)
    return micro_avg_f1,micro_avg_prec,micro_avg_rec

def macro_scores(y_true,y_pred):
    precesion,recall,_,_,_ = precesion_recall(y_true,y_pred)
    prec_avg =round(float(np.nansum(precesion)/(len(precesion))),3)
    rec_avg = round(float(np.nansum(recall)/len(recall)),3)
    macro_f1 = round((2*prec_avg*rec_avg)/(prec_avg+rec_avg),3)
    return macro_f1,prec_avg,rec_avg

def save_image(df,table_name):
    print(df)
    fig, ax = plt.subplots(figsize=(20, 2)) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, df, loc='upper right', colWidths=[0.1]*len(df.columns))  # where df is your data frame
    tabla.auto_set_font_size(False) # Activate set fontsize manually
    tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
    tabla.scale(1.3, 1.3) # change size table
    plt.savefig(table_name)


#%%
if __name__ == "__main__":
#    input_filename = sys.argv[1]
#    test_file = sys.argv[2]
#    param_text = sys.argv[3]
#    output_file = sys.argv[4]
#    weights_file =sys.argv[5]
   input_filename = "train.csv"
   test_file = "test_X.csv"
   param_text = "param_a.txt"
   output_file = "out.txt"
   weights_file ="weigh.txt"
   
   param_file = open(param_text,'r')
   param =[]
   for line in param_file:
       param.append(line.strip())


#    df_train = pd.read_csv(input_filename,header=None)
#    df_test = pd.read_csv(test_file,header=None)
#%%
   trainX,trainY,testX,mapping_class = read_inputs(input_filename,test_file)
   class_weights = class_weight.compute_class_weight("balanced",np.unique(trainY),trainY)
   Y_true = decode(trainY,mapping_class)
   print(testX.shape)
   trainY = onehotEncoder(trainY,len(np.unique(trainY)))
   trainX_encoded = np.ones((trainX.shape[0],1))
   testX_encoded = np.ones((testX.shape[0],1))
   for i in range(1,trainX.shape[1]):
        n_k = len(np.unique(trainX[:,i]).tolist())
        enc = onehotEncoder(trainX[:,i],n_k)
        trainX_encoded= np.hstack((trainX_encoded,enc))
   for i in range(1,testX.shape[1]):
        n_k = len(np.unique(testX[:,i]).tolist())
        enc = onehotEncoder(testX[:,i],n_k)
        testX_encoded= np.hstack((testX_encoded,enc))
   print(testX_encoded.shape)

#%%
   if(param[0]=="1"):
        lr = float(param[1])
        itera = int(param[2])
        #batch_size=int(param[3])
        cost_hist,theta,grad = run_model(trainX_encoded,trainY,lr,itera,adaptive=False,class_weights=class_weights)
   if(param[0]=="2"):
       lr = float(param[1])
       itera = int(param[2])
       #batch_size=int(param[3])       
       cost_hist,theta,grad = run_model(trainX_encoded,trainY,lr,itera,adaptive=True,class_weights=class_weights)
   if(param[0]=='3'):
       albe = param[1].split(",")
       lr = float(albe[0])
       alpha = float(albe[1])
       beta = float(albe[2])
       itera = int(param[2])
       #batch_size = int(param[3])
       cost_hist,theta,grad = run_model(trainX_encoded,trainY,lr,itera,alpha=alpha,Beta=beta,class_weights=class_weights)
#%%
   pred_test,pred_test_encoded = prediction(theta,testX_encoded,mapping_class)
   pred_train,pred_train_encoded = prediction(theta,trainX_encoded,mapping_class)
   accuracy = np.sum([1 if x==y else 0 for (x,y) in zip(pred_train,Y_true.ravel())])
   print(float(accuracy/Y_true.shape[0]))
   train_pred = onehotEncoder(pred_train_encoded.ravel(),5)
   confusionmatrix = confusion_matrix(trainY,train_pred)
   precesion,recall,tp,fp,fn = precesion_recall(trainY,train_pred)

   micro_f1,avg_prec_micro,avg_rec_micro = micro_scores(trainY,train_pred)
   macro_f1,avg_prec_macro,avg_rec_macro = macro_scores(trainY,train_pred)

   print(confusionmatrix)
   with open(output_file,"w") as f:
       f.write("\n".join(x for x in pred_test))

   with open(weights_file,"w") as f:
       listtheta =theta.tolist()
       for line in listtheta:
            i=0
            for i in range(len(line)):
                  f.write(str(line[i]))
                  if(i<=len(line)-2):
                        f.write(",")
            f.write("\n")

   plt.plot(cost_hist)
   plt.show()








#%%
f1_score_per_class = np.around(((2*np.array(precesion)*np.array(recall))/(np.array(precesion)+np.array(recall))),decimals=2).reshape((5,1))
confusionmatrix = np.hstack((confusionmatrix,np.array(precesion).reshape((5,1)),np.array(recall).reshape((5,1)),f1_score_per_class))
df_per_class = pd.DataFrame(confusionmatrix,columns=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior',"precesion","recall","F1_scores"],index=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])
save_image(df_per_class,"confusion_table_"+param[0]+"_case.png")


df_micro_macro = pd.DataFrame(np.array([[micro_f1,macro_f1],[avg_prec_micro,avg_prec_macro],[avg_rec_macro,avg_rec_micro]]),columns=["Micro Scores","Macro Scores"],index=["F1","average precall","average recall"])
save_image(df_micro_macro,"scores_table_"+param[0]+"_case_lr_"+param[1]+"_iter_"+param[2]+"_.png")

plt.figure(figsize=(10,10))