# -*- coding: utf-8 -*-


import numpy as np


inFile2=open('airfoil_self_noisedat.txt')

Xairfoil=[];Yairfoil=[]; # initializing theta,X, and Y vectors
theta=np.ones(6,dtype='float64');
theta=np.matrix(theta)
theta=theta.T


for i,line in enumerate(inFile2.readlines()):

    Xairfoil.append(line.strip().split('\t'))
    Yairfoil.append(Xairfoil[i].pop())
    

 # adding the X0's as 1
for i in Xairfoil:
    i.insert(0,1)


Xairfoil=np.array(Xairfoil,dtype='float64');XX=Xairfoil.copy();
Xairfoil = Xairfoil / Xairfoil.max(axis=0) # normalizes features
Yairfoil=np.array(Yairfoil,dtype='float64')
Yairfoil=np.matrix(Yairfoil);
Yairfoil=Yairfoil.T;

# hyperparameters
m=len(Xairfoil) #number of training examples
alpha=1.05; # learning rate
iter=1000;  #number of iterations for convergence

#gradient descent
for i in range(iter):
    theta-=(alpha/m)*(np.matmul(Xairfoil.T,(np.matmul(Xairfoil,theta)-Yairfoil)))
    
#prints trained theta 
print(theta)

# a previously trained networks weights with learning rate = 1.05, iterations=1000
thetaConverged=np.matrix([[ 132.83370499],
 [ -25.64401798],
 [  -9.3660757 ],
 [ -10.87757756],
 [   7.11956223],
 [  -8.60434706]])
    

# predict a given x value given its index in Xairfoil and the theta vector(1-D np.matrix)
def predict(index,thetaConverged):
    print(np.matmul(np.matrix(Xairfoil[index]),thetaConverged))
    

#Cost function
hFunc=np.matmul(Xairfoil,theta)
term=hFunc-Yairfoil
J=(1/(2*m))*(np.matmul(term.T,term))
print(f'The cost function J is {J}')



