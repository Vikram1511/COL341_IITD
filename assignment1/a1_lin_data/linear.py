import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LassoLars 

import sys
import time
import tqdm
from tqdm import tqdm
from itertools import combinations


import warnings
warnings.filterwarnings("ignore")

mode = sys.argv[1]
file_train = sys.argv[2]
file_test = sys.argv[3]

def normalEquation(X,Y):
    term1 = np.matmul(np.transpose(X),X)
    term2 = np.matmul(np.transpose(X),Y)
    opt_weights = np.matmul(np.linalg.inv(term1),term2)
    return opt_weights

def ridge_normalEquation(X,Y,lamda):
    term1 = np.matmul(np.transpose(X),X)
    term2 = np.dot(lamda, np.eye(term1.shape[0]))
    term3 = np.matmul(np.transpose(X),Y)
    opt_weights = np.matmul(np.linalg.inv(term1+term2),term3)
    return opt_weights

def predictions(X,theta):
    assert X.shape[1] == theta.shape[0]
    predict = np.matmul(X,theta)
    return predict

def run_model(X,Y):
    opt_theta = normalEquation(X,Y)
    loss = loss_func(X,Y,opt_theta)
    return opt_theta,loss

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

def loss_func(X,Y,W,lamda=None):
    n = X.shape[0]
    term1 = np.matmul(X,W) - Y
    #print(W.shape)
    term1 = (np.matmul(np.transpose(term1),term1))/(n)
    # if(lamda):
    #     term2 = np.matmul(np.transpose(W),W)
    #     term2 = (term2*lamda)/2
    term2 = np.sum(np.power((Y-np.average(Y,axis=0)),2),axis=0)/n
    loss  = term1.item()/term2.item()

    return loss

def lassolars(X,Y,lamda):
    model = LassoLars(alpha=lamda)
    model.fit(X,Y)
    return model

def cross_validation_regularization(X,Y,K,lamda,lasso=False):
    Kfolds  = kfolds(X,Y,K)
    cross_val = []

    for i in range(K):
        #vertical stacking of other than validation set of kfolds array
        train_set = Kfolds[:i]+Kfolds[i+1:]
        train_set = np.vstack((train_set))

        #trainX and trainY distinguishing
        trainX = train_set[:,:train_set.shape[1]-1]
        trainY = train_set[:,train_set.shape[1]-1].reshape((train_set.shape[0],1))

        #validation set 
        val_set = Kfolds[i]

        #valx and valy distinguishing
        valX = val_set[:,:val_set.shape[1]-1]
        valY = val_set[:,val_set.shape[1]-1].reshape((val_set.shape[0],1))
        
        #optimum weights from ridge normal equation
        if(lasso==True):
            Model = lassolars(trainX,trainY,lamda)
            weights = Model.coef_
            weights = weights.reshape((weights.shape[0],1))
        else:
            weights = ridge_normalEquation(trainX,trainY,lamda)

        #calculating loss value over validation set
        loss_val = loss_func(valX,valY,weights)

        #appending this loss value in cross_val array
        cross_val.append(loss_val)

    return cross_val

        




def read_inputs(trainFile,testFile):
    df_train = pd.read_csv(trainFile,header=None)
    no_of_columns = len(df_train.columns)

    df_test = pd.read_csv(testFile,header=None)

    Y = df_train[no_of_columns-1].values.reshape((df_train.shape[0],1))
    #print(Y.shape)
    df_train.drop(no_of_columns-1,axis=1,inplace=True)

    X = df_train.values
    X = np.hstack((np.ones((X.shape[0],1)),X))
    X_test = df_test.values 
    X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))
    return X,Y,X_test

def polynomial_set(X,power_n=None,comb_n = None,iscombination=False,ispoly=False):
    # assert ispoly==True and power_n!=None
    # assert iscombination==True and comb_n!=None

    n,m = X.shape
    if(ispoly==True):
        for i in range(m):
            X= np.hstack((X,np.power(X[:,i],power_n).reshape((X.shape[0],1))))
    
    if(iscombination==True):
        comb = combinations(range(m))
        for item in list(comb):
                comb_array = np.ones((n,1))
                item_n = len(item)
                for i in range(item_n):
                    comb_array = np.dot(comb_array,X[item[i]])
                X  = np.hstack((X,comb_array))  
    return X



def my_print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

if(mode=="a"):
    #output.txt and weights.txt files input
    file_output = sys.argv[4]
    file_weight = sys.argv[5]

    #reading dataset files
    trainX,trainY,testX = read_inputs(file_train,file_test)

    #getting optimum weights via normal equation running model
    optTheta,loss = run_model(trainX,trainY)
    #print(loss)
    #getting predictions
    prediction = predictions(testX,optTheta)
    with open(file_output,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in prediction))
    
    with open(file_weight,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in optTheta))

if(mode=="b"):
    lamda_file = sys.argv[4]
    file_output = sys.argv[5]
    file_weight = sys.argv[6]

    trainX,trainY,testX = read_inputs(file_train,file_test)
    lamda_array = []
    lamdas = []
    lamda_file = open(lamda_file,"r")
    for line in lamda_file:
        lamda_array.append(line.strip())
    
    for x in lamda_array:
        try:
            lamdas.append(float(x))
        except:
            pass
    
    loss_values=[]
    opt_lamda=0
    cv_loss_opt = np.inf

    for i in tqdm(range(len(lamdas))):
        cv = cross_validation_regularization(trainX,trainY,10,lamdas[i])
        mean_cv_loss = sum(cv)/len(cv)
        loss_values.append({'mean':mean_cv_loss,'minimum':min(cv),'maximum':max(cv)})
        if(mean_cv_loss < cv_loss_opt):
            cv_loss_opt = mean_cv_loss
            opt_lamda = lamdas[i]
        time.sleep(2)

    plotData=[]
    for i  in range(len(loss_values)):
        plotData.append(loss_values[i]['mean'])
    
    plt.plot(lamdas,plotData,"b+")
    plt.xlabel("lamda")
    plt.ylabel("average MSE cross validation loss")
    plt.title("Cross Validation")
    plt.savefig("lamda_vs_cvloss")

    print(opt_lamda)
    #print("average loss for cross_validation:",cv_loss_opt)
    #print(len(loss_values))
    #rint(loss_values[0])

    optTheta = ridge_normalEquation(trainX,trainY,opt_lamda)
    prediction = predictions(testX,optTheta)
    with open(file_output,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in prediction))
    
    with open(file_weight,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in optTheta))

if(mode=="c"):
    file_output = sys.argv[4]
    trainX,trainY,testX = read_inputs(file_train,file_test)

    #trainX = polynomial_set(trainX,power_n = 2,ispoly=True)
    #testX  = polynomial_set(testX,power_n = 2,ispoly=True)
    lamdas = [0.0001,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 100]

    loss_values=[]
    opt_lamda=0
    cv_loss_opt = np.inf
    for i in tqdm(range(len(lamdas))):
        cv = cross_validation_regularization(trainX,trainY,10,lamdas[i],lasso=True)
        mean_cv_loss = sum(cv)/len(cv)
        loss_values.append({'mean':mean_cv_loss,'minimum':min(cv),'maximum':max(cv)})
        if(mean_cv_loss < cv_loss_opt):
            cv_loss_opt = mean_cv_loss
            opt_lamda = lamdas[i]
        time.sleep(2)
    
    plotData=[]
    for i  in range(len(loss_values)):
        plotData.append(loss_values[i]['mean'])
    
    plt.plot(lamdas,plotData,"b+")
    plt.xlabel("lamda")
    plt.ylabel("average MSE cross validation loss(LassoRegression)")
    plt.title("Cross Validation")
    plt.savefig("lamda_vs_CrossValidationLoss")

    print(opt_lamda)
    print(cv_loss_opt)
    model = LassoLars(alpha=opt_lamda)
    model.fit(trainX,trainY)
    prediction = model.predict(testX)
    prediction = prediction.reshape((prediction.shape[0],1))
    with open(file_output,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in prediction))
    


