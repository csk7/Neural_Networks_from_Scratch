# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 19:48:37 2017

@author: Siva
"""
import numpy as np
#import matplotlib.pyplot as plt
import gzip,pickle,pylab
import time
logFile=r'C:\Users\Siva\Desktop\logFile.log'
f=gzip.open(r'C:\Users\Siva\Desktop\mnist.pkl.gz')
MNIST=pickle.load(f)

MOONS = np.loadtxt(open(r'C:\Users\Siva\Desktop\2moons.txt','r'))
    
def RELU(data):
    return np.maximum(data, 0)

def RELU_der(data):
    data_out=data
    data_out[[data<=0]]=0
    data_out[[data>0]]=1
    return data_out
    
def sigmoid(data):
  return 1 / (1 + np.exp(-data))

def softmax(data):
    o_exp = np.exp(data - np.max(data))
    if np.size(data)==1:
        return o_exp / np.sum(o_exp)
    else:
        return o_exp / np.sum(o_exp, axis = 0, keepdims = True)
    
def mnist_fun(data):
    data_out=data
    data_out[[data==0]]=0
    data_out[[data!=0]]=1
    return data_out
    
class NeuralNetwork(object):
    def __init__(self,m,d_h,d,l21,l22,l11,l12,k):
        self.m=m
        self.d=d
        self.d_h=d_h
        self.weight1=(np.random.rand(self.d_h,self.d)-0.5)*(2/np.sqrt(self.d))
        self.bias1=np.zeros((self.d_h,1))
        self.weight2=(np.random.rand(self.m,self.d_h)-0.5)*(2/np.sqrt(self.d_h))
        self.bias2=np.zeros((self.m,1))
        self.l21=l21
        self.l22=l22
        self.l11=l11
        self.l12=l12
        self.K = k
        
    def computePredictions(self,testSet):
        #initialise prediction array
        predictions=np.zeros((np.shape(testSet)[0],1))
        self.loss_cp=np.zeros((np.shape(testSet)[0],1))
        for i in range(np.shape(testSet)[0]):
             h_a=np.dot(self.weight1,testSet[i,:].reshape(-1,1))+self.bias1
             #apply RELU
             h_s=RELU(h_a)
             #calculation of o_a
             o_a=np.dot(self.weight2,h_s)+self.bias2
             #calculation of o_s ufing softmax
             o_s=softmax(o_a)
             predictions[i,0] = np.argmax(o_s)
              
             self.loss_cp[i,0]=(-np.log(o_s[int(np.argmax(o_s)),0]) + self.l21*np.sum(np.absolute(self.weight2)) +  self.l22*np.sum(np.square(self.weight2)) + self.l11*np.sum(np.absolute(self.weight1)) +  self.l12*np.sum(np.square(self.weight1)))
        return predictions            

    def dofrop_k(self,x,target):
        self.x_reshaped = np.transpose(x)
        self.target_k = target
        self.h_a_k = np.dot(self.weight1,self.x_reshaped) + self.bias1
        self.h_s_k = RELU(self.h_a_k)
        
        
        self.o_a_k = np.dot(self.weight2,self.h_s_k) + self.bias2
        self.o_s_k = softmax(self.o_a_k)
        
        self.Loss_k = np.zeros((1,self.K))
        for j in range(self.K):
            #print self.o_s_k[int(self.target_k[j]),j]
            self.Loss_k[0,j] = -np.log(self.o_s_k[int(self.target_k[j]),j] + self.l21*np.sum(np.absolute(self.weight2)) +  self.l22*np.sum(np.square(self.weight2)) + self.l11*np.sum(np.absolute(self.weight1)) +  self.l12*np.sum(np.square(self.weight1)))         
        
           
# Matrix Method Calculation    
    def dobprop_k(self):
        self.onehot_k=np.zeros((self.m,self.K))
        for j in range(self.K):
            self.onehot_k[int(self.target_k[j]),j]=1
        self.grad_oa_k=self.o_s_k-self.onehot_k        

        self.grad_b2_k=self.grad_oa_k
        self.grad_W2_k=np.zeros((self.K,self.m,self.d_h))
        for j in range(self.K):
            self.grad_W2_k[j,:,:]=np.dot(self.grad_oa_k[:,j].reshape(-1,1),self.h_s_k[:,j].reshape(1,-1)) + (self.l21 * np.sign(self.weight2)) + 2* self.l22*self.weight2

            
            
        self.grad_hs_k=np.zeros((self.d_h,self.K))        
        for j in range(self.d_h):
             self.grad_hs_k[j,:] = np.dot(np.transpose(self.weight2[:,j]),self.grad_oa_k)
             
        self.grad_ha_k =self.grad_hs_k * RELU_der(self.h_s_k)
             
        self.grad_b1_k = self.grad_ha_k
        self.grad_W1_k=np.zeros((self.K,self.d_h,self.d)) 

        for j in range(self.K):
            self.grad_W1_k[j,:,:] = np.dot(self.grad_ha_k[:,j].reshape(-1,1),self.x_reshaped[:,j].reshape(1,-1)) + (self.l11*np.sign(self.weight1)) + 2* self.l12*self.weight1
        
        self.grad_x_k=np.zeros((self.d,self.K)) 
        for j in range(self.d):
           self.grad_x_k = np.dot(np.transpose(self.weight1[:,j]), self.grad_ha_k)
            
            
    def trainModel_matrix(self,data_set,valid_set,test_set,iterations,h):
        log=open(logFile,'w')
        self.data_set=data_set
        self.valid_set=valid_set
        self.test_set=test_set
        self.maxIterations=iterations
        self.step=h
        self.lengthData=self.data_set.shape[0]
        iteration=0
        tr_ep=np.zeros((iterations,1))
        tr_loss=np.zeros((iterations,1))
        v_ep=np.zeros((iterations,1))
        t_ep=np.zeros((iterations,1))
        v_loss=np.zeros((iterations,1))
        t_loss=np.zeros((iterations,1))        
        while iteration<self.maxIterations:
            k_index=0           
            epoch_loss=0.0
            while k_index<(self.lengthData - self.K):
                
                self.miniBatchLoss=np.zeros((1,1))
                self.miniBatchGrad_W1=np.zeros((self.d_h,self.d))
                self.miniBatchGrad_b1=np.zeros((self.d_h,1))
                self.miniBatchGrad_W2=np.zeros((self.m,self.d_h))
                self.miniBatchGrad_b2=np.zeros((self.m,1))

     
                self.dofrop_k(self.data_set[k_index:(k_index + self.K),:-1],self.data_set[k_index:(k_index + self.K),-1])
                self.dobprop_k();
                self.miniBatchLoss=np.sum(self.Loss_k,axis=1)
                self.miniBatchGrad_W1=np.sum(self.grad_W1_k,axis=0)              
                self.miniBatchGrad_b1=np.sum(self.grad_b1_k,axis=1).reshape(-1,1)
                self.miniBatchGrad_W2=np.sum(self.grad_W2_k,axis=0)
                self.miniBatchGrad_b2=np.sum(self.grad_b2_k,axis=1).reshape(-1,1)             
                
                
                k_index = k_index + self.K    
                risk=(self.miniBatchLoss)/self.K;
                epoch_loss=epoch_loss + risk                
                self.weight1=self.weight1-self.step*self.miniBatchGrad_W1/self.K
                self.weight2=self.weight2-self.step*self.miniBatchGrad_W2/self.K
                self.bias1=self.bias1-self.step*self.miniBatchGrad_b1/self.K
                self.bias2=self.bias2-self.step*self.miniBatchGrad_b2/self.K
                #predictions=testModel.computePredictions(self.data_set[:,:-1])
                #ep= (np.sum(np.abs(predictions-(self.data_set[:,-1]).reshape(-1,1)))/self.lengthData)*100
                #print  "training error percent" ,  ep     
                #print "Epoch :",iteration ,"Loss of Mini batch is :",risk
            
            predictions=testModel.computePredictions(self.data_set[:,:-1])
            tr_ep[iteration,0]= ((np.sum(mnist_fun(predictions-self.data_set[:,-1].reshape(-1,1))))/self.lengthData*100) 
            log.write("\n \nEpoch : " +str(iteration) + " training error percent " + str(tr_ep[iteration,0]) + " %")
            print "\n \nEpoch :",iteration , " training error percent" , tr_ep[iteration,0]
            tr_loss[iteration,0] = (np.sum(self.loss_cp)/self.lengthData)                                    
            log.write("\nEpoch : "+ str(iteration) + " Training Loss is :" + str(tr_loss[iteration,0]))
            print "Epoch :",iteration ," Training Loss is :",tr_loss[iteration,0]
            
            predictions=testModel.computePredictions(self.valid_set[:,:-1])
            v_ep[iteration,0]= (np.sum(mnist_fun(predictions-self.valid_set[:,-1].reshape(-1,1))))/np.shape(self.valid_set)[0]*100
            log.write("\nEpoch : " + str(iteration) + " Validation error percent "  + str(v_ep[iteration,0]) + " %")
            print "Epoch :",iteration , " Validation error percent" , v_ep[iteration,0]
            v_loss[iteration,0] = np.sum(self.loss_cp)/(np.shape(self.valid_set)[0])
            log.write("\nEpoch : "+str(iteration) + " Validation Loss is :" + str(v_loss[iteration,0]))
            print "Epoch :",iteration ," Validation Loss is :", v_loss[iteration,0]
            #print "Epoch :",iteration ,"Loss is :",epoch_loss
            
            predictions=testModel.computePredictions(self.test_set[:,:-1])
            t_ep[iteration,0]= (np.sum(mnist_fun(predictions-self.test_set[:,-1].reshape(-1,1))))/np.shape(self.test_set)[0]*100
            log.write("\nEpoch : " + str(iteration) + " Test error percent " +  str(t_ep[iteration,0]) + " %")
            print "Epoch :",iteration , " Test error percent" ,  t_ep[iteration,0] 
            t_loss[iteration,0] = np.sum(self.loss_cp)/(np.shape(self.test_set)[0])
            log.write("\nEpoch : "+ str(iteration) +" Test Loss is :"+ str(t_loss[iteration,0]))
            print "Epoch :",iteration ," Test Loss is :", t_loss[iteration,0]
            #print "Epoch :",iteration ,"Loss is :",epoch_loss
            
            iteration = iteration + 1
        
        log.close()
        pylab.figure(1)    
        pylab.ylabel('Error %')
        pylab.xlabel('Iterations')
        x=np.linspace(0,self.maxIterations,self.maxIterations)
        h1, = pylab.plot(x,tr_ep[:,0],'-b', label = 'Train')
        h2, = pylab.plot(x,v_ep[:,0],'-r', label = 'Valid')
        h3, = pylab.plot(x,t_ep[:,0],'-g', label = 'Test') 
        pylab.legend(handles=[h1, h2, h3])
        
        pylab.figure(2) 
        pylab.ylabel('Loss')
        pylab.xlabel('Iterations')
        x=np.linspace(0,self.maxIterations,self.maxIterations)
        f1, = pylab.plot(x,tr_loss[:,0],'-b', label = 'Train')
        f2, = pylab.plot(x,v_loss[:,0],'-r', label = 'Valid')
        f3, = pylab.plot(x,t_loss[:,0],'-g', label = 'Test')
        pylab.legend(handles=[f1, f2, f3])        
          
            
print "\n*******************QUESTION 3.9 ,3.10***************************\n"
d=784
m=10
d_h=30
k=50

testModel=NeuralNetwork(m,50,d,0.00006,0.00005,0.00004,0.00005,50)
train_set = np.append(MNIST[0][0],MNIST[0][1].reshape(-1,1),axis=1)
valid_set = np.append(MNIST[1][0],MNIST[1][1].reshape(-1,1),axis=1)
test_set = np.append(MNIST[2][0],MNIST[2][1].reshape(-1,1),axis=1)

testModel.trainModel_matrix(train_set,valid_set,test_set,40,0.2)

       
test_predictions=testModel.computePredictions(test_set[:,:-1])
t_ep= (np.sum(mnist_fun(test_predictions-(test_set[:,-1]).reshape(-1,1)))/np.shape(test_set)[0])*100    
print  "\n Test error percent" ,  t_ep