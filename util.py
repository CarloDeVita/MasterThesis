import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
import cv2
import tensorflow as tf
import scipy as sc
from scipy import io
import pickle
import dill
import pandas as pnd

def saveFile(obj,namefile):
    with open(namefile,"wb") as fp:
        dill.dump(obj,fp)

def loadFile(namefile):
    with open(namefile,"rb") as fp:
        obj=dill.load(fp)
    return obj

def takeDictEachNumber(dataset, struttura):
    
    dicts=[]
    for i in range(10):
        diz=loadFile(f"Dizionari/diz{dataset}_{i} {struttura}.dump")
        dicts.append(diz)
    return dicts

def relCoeffs(number,coeffs,relevance):
    ret = np.zeros((28,28))
    for i in range(number):
        ret = abs(coeffs[:,:,0,0,relevance[i]]) + ret        
    return ret

def relCoeffsResized(number,coeffs,relevance):
    ret = np.zeros((64,64))
    for i in range(number):
        ret = abs(coeffs[:,:,0,0,relevance[i]]) + ret        
    return ret

def relConvs(number,cnv,relevance):
    ret=np.zeros((28,28))
    for i in range(number):
        ret=np.dstack((cnv[relevance[i]],ret))
    ret=np.delete(ret,-1,2)
    ret=np.flip(ret,2)
    return ret

def relConvsResized(number,cnv,relevance):
    ret=np.zeros((64,64))
    for i in range(number):
        ret=np.dstack((cnv[relevance[i]],ret))
    ret=np.delete(ret,-1,2)
    ret=np.flip(ret,2)
    return ret

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


#def loadDataAtt():
#    k=0;
#    img=np.zeros((400,64,64))
#    for i in range(40):
#        j=0
#        for j in range(10):
#            x = image.imread(f'attFaces/s{i+1}/{j+1}.pgm')
#            x=x/255
#            #resize and contrast normalize
#            x = cv2.resize(x,(64,64))  
#            blurred = cv2.blur(x,(5,5))
#            img[k] = x - blurred
#            k=k+1
#            #plt.show()
#            
#    return img
        
        
def loadJaffeCropped():
    data=sc.io.loadmat('jaffeCropped.mat')
    x_train=data['Xtrain_shuffled']
    x_train=x_train.reshape(32,32,213)
    x_train=cv2.resize(x_train,(64,64))
    x_train=x_train.transpose(2,1,0)
    y_train=data['Ytrain_shuffled']
    
    labels=np.zeros(213)
    b_train=np.zeros((213,64,64))
    for i in range(213):
        blurred=cv2.blur(x_train[i],(3,3))
        b_train[i]=x_train[i]-blurred
        z=y_train[:,i]
        r=np.where(z==1)
        label=r[0][0]
        labels[i]=label
    labels=labels.astype(int)
    return x_train,b_train,labels

def loadJaffeCroppedToolbox():
    data=sc.io.loadmat('jaffeCropped.mat')
    x_train=data['Xtrain_shuffled']
    x_train=x_train.reshape(32,32,213)
    x_train=cv2.resize(x_train,(64,64))
    x_train=x_train.transpose(2,1,0)
    y_train=data['Ytrain_shuffled']
    
    labels=np.zeros(213)
    b_train=np.zeros((213,64,64))
    for i in range(213):
        blurred=cv2.blur(x_train[i],(5,5))
        b_train[i]=x_train[i]-blurred
        z=y_train[:,i]
        r=np.where(z==1)
        label=r[0][0]
        labels[i]=label
    labels=labels.astype(int)
    return x_train,b_train,labels

def loadJaffeCropped_orig():
    data=sc.io.loadmat('jaffeCropped.mat')
    x_train=data['Xtrain_original']
    x_train=x_train.reshape(96,128,213)
#    x_train=cv2.resize(x_train,(32,32))
    x_train=x_train.transpose(2,1,0)
    y_train=data['Ytrain']
    
    labels=np.zeros(213)
    b_train=np.zeros((213,128,96))
    for i in range(213):
        blurred=cv2.blur(x_train[i],(7,7))
        b_train[i]=x_train[i]-blurred
        z=y_train[:,i]
        r=np.where(z==1)
        label=r[0][0]
        labels[i]=label
    labels=labels.astype(int)
    return x_train,b_train,labels   
    
def loadJaffe():
    data=sc.io.loadmat('jaffe32x32.mat')
    return data


def loadDataAtt():
    k1=0;
    k2=0
    dim=64
    S=np.zeros((360,dim,dim))
    B=np.zeros((360,dim,dim))
    y_train=np.zeros(360)
    Stest=np.zeros((40,dim,dim))
    Btest=np.zeros((40,dim,dim))
    y_test=np.zeros(40)
    for i in range(40):
        j=0
        for j in range(10):
            x = image.imread(f'attFaces/s{i+1}/{j+1}.pgm')
            x=x/255
#            print(x.shape)
            #resize and contrast normalize
            x = cv2.resize(x,(dim,dim))  
            blurred = cv2.blur(x,(5,5))
            if(j < 9):
                S[k1]=x;
                B[k1] = x - blurred
                y_train[k1]=i
                k1=k1+1
            else:
                Stest[k2]=x
                Btest[k2]=x-blurred
                y_test[k2]=i
                k2=k2+1
            #plt.show()
        
    y_train=y_train.astype(int)
    y_test=y_test.astype(int)
                           
    return S,B,y_train,Stest,Btest,y_test

def loadMNIST(num=100):
    mnist = tf.keras.datasets.mnist;
    (S, y_train), (Stest, y_test) = mnist.load_data()
    
    S=S/255
    nimgs = np.size(S,0)
    B = np.zeros(S.shape)
    for i in range(nimgs):
        blurred = cv2.blur(S[i],(3,3))
        B[i] = S[i] - blurred
    
    Stest = Stest/255
    nimgs = np.size(Stest,0)
    Btest = np.zeros(Stest.shape)
    for i in range(nimgs):
        blurred = cv2.blur(Stest[i],(3,3))
        Btest[i] = Stest[i] - blurred
#        Btest[i]=Btest[i]-Btest[i].min()
#        Btest[i]=Btest[i]/Btest[i].max()
    return S,B,y_train,Stest,Btest,y_test


def loadMNIST_resized(num=100):
    mnist = tf.keras.datasets.mnist;
    (S1, y_train), (S1test, y_test) = mnist.load_data()
    
    S1=S1/255
    nimgs = np.size(S1,0)
    S= np.zeros((nimgs,64,64))
    B = np.zeros(S.shape)
    for i in range(nimgs):
        S[i]=cv2.resize(S1[i],(64,64))
        blurred = cv2.blur(S[i],(5,5))
        B[i] = S[i] - blurred
    
    S1test = S1test/255
    nimgs = np.size(S1test,0)
    Stest= np.zeros((nimgs,64,64))
    Btest = np.zeros(Stest.shape)
    for i in range(nimgs):
        Stest[i]=cv2.resize(S1test[i],(64,64))
        blurred = cv2.blur(Stest[i],(5,5))
        Btest[i] = Stest[i] - blurred
    return S,B,y_train,Stest,Btest,y_test

def loadFashionMNIST(num=100):
    mnist = tf.keras.datasets.fashion_mnist;
    (S, y_train), (Stest, y_test) = mnist.load_data()
    
    S=S/255
    nimgs = np.size(S,0)
    B = np.zeros(S.shape)
    for i in range(nimgs):
        blurred = cv2.blur(S[i],(3,3))
        B[i] = S[i] - blurred
    
    Stest = Stest/255
    nimgs = np.size(Stest,0)
    Btest = np.zeros(Stest.shape)
    for i in range(nimgs):
        blurred = cv2.blur(Stest[i],(3,3))
        Btest[i] = Stest[i] - blurred
    return S,B,y_train,Stest,Btest,y_test
    

def groupMNIST(X,Y):
    l0=groupMNIST2(0,X,Y) 
    l1=groupMNIST2(1,X,Y)
    l2=groupMNIST2(2,X,Y)
    l3=groupMNIST2(3,X,Y)
    l4=groupMNIST2(4,X,Y)
    l5=groupMNIST2(5,X,Y)
    l6=groupMNIST2(6,X,Y)
    l7=groupMNIST2(7,X,Y)
    l8=groupMNIST2(8,X,Y)
    l9=groupMNIST2(9,X,Y)
    return l0,l1,l2,l3,l4,l5,l6,l7,l8,l9
    
    
    
    
def groupMNIST2(label,X,Y):
    nimgs=np.size(X,0)
    l=np.zeros((28,28));
    for i in range(nimgs):
        if(Y[i]==label):
            l=np.dstack((l,X[i]))
    l=np.delete(l,-1,2)
    return l


    
def loadGabJaffe():
    data=sc.io.loadmat('Gabor_jaffe.mat')
    return data
    
        
def loadNN_mnist():
    data = sc.io.loadmat('long-rect.mat')
    return data

def loadMNISTtoolbox():
    mnist = tf.keras.datasets.mnist;
    (S, y_train), (Stest, y_test) = mnist.load_data()
    S=np.resize(S,((60000,784)))
    Stest=np.resize(Stest,((10000,784)))
    ytr=np.zeros((60000,10))
    for i in range(60000):
        y=y_train[i]
        ytr[i][y]=1
    

    yts=np.zeros((10000,10))  
    for i in range(10000):
        y=y_test[i]
        yts[i][y]=1
    return S, ytr, Stest, yts
    
