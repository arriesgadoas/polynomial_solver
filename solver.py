#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:40:22 2022

@author: ali arriesgado
"""

from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randrange


#open training data set and store as a list
with open('data_train.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
    
#function for min-max normalization
# def normalize(x,x_min,x_max):
#     x_ = (x-x_min)/(x_max-x_min)
#     return x_

def normalize(x,x_mean,x_std):
    x_ = (x-x_mean)/(x_std)
    return x_

#function to convert 1d vector to 2d array
def one2twoD(x):
    m = np.shape(x)[0]
    x_ = np.reshape(x,(m,1))
    return x_

#function to get the squared error
def se(target, prediction):
    error = (target - prediction)**2
    return error

#function to get r-squared
def r_squared(target, prediction):
    
    r_square = 1- ((target-prediction)**2).sum() / ((target-target.mean())**2).sum()
    return r_square

#define my model, single layer, one node and 6 weights
class my_model:
  def __init__(self):
    self.l1 = Tensor.zeros(5,1)
    
  def forward(self, x):
    return x.dot(self.l1).relu()

op = np.array(data)

#remove column headers and convert string to float
op = op[1:].astype(float)

#shuffle training data for SGD
#np.random.shuffle(op)

#separate feature and label
x,y = op.T

#get sample size
m = np.shape(x)[0]

#convert 1D array to 2D array for operations
xs = np.reshape(x,(m,1))
ys = np.reshape(y,(m,1))

#plot
# plt.figure()
# plt.plot(xs,ys)
# plt.xlabel('x') 
# plt.ylabel('y') 
# plt.title("Plot of training data")
# plt.show()

#store min and max values to use for feature scaling in the test data
x1_mean = np.mean(xs)
x1_std = np.std(xs)
x2_mean = np.mean(xs**2)
x2_std = np.std(xs**2)
x3_mean = np.mean(xs**3)
x3_std = np.std(xs**3)
x4_mean = np.mean(xs**4)
x4_std = np.std(xs**4)

#feature scaling of the fifth order polynomial
x1 = one2twoD(normalize(xs,x1_mean,x1_std)) 
x2 = one2twoD(normalize(xs**2,x2_mean,x2_std))
x3 = one2twoD(normalize(xs**3,x3_mean,x3_std))
x4 = one2twoD(normalize(xs**4,x4_mean,x4_std))
#x5 = one2twoD(normalize(xs**5))

#X = np.concatenate((np.ones((m,1)), x1, x2, x3, x4, x5),axis = 1)
X = np.concatenate((np.ones((m,1)), x1, x2, x3, x4),axis = 1)
#convert numpy arrays to tinygrad tensors
ysT = Tensor(ys,requires_grad=True)

XT = Tensor(X,requires_grad=True)

model = my_model()
model.l1 = Tensor.uniform(5,1)
optim = optim.SGD([model.l1], lr=0.001)


tolerance = 1e-3
epochs = 15
break_out_flag = False

for i in range(epochs+1):
   # optim.zero_grad()
    for j in range(m):
        k = randrange(m)
       # optim.zero_grad()
        out = model.forward(XT[k])
        optim.zero_grad()
        loss = se(ysT[k], out)
        loss.backward()
        optim.step()

        if(abs((out-ysT[k]).data) <= tolerance):
            print(f"stop: {(out-ysT[k]).data}")
            # break_out_flag = True
            break
            
    if break_out_flag:
            break 
            
    if (i % 5) == 0:
        print(f"Epoch: {i}/{epochs} ---> R-squared: {r_squared(ysT, XT.matmul(model.l1)).data}")

#print(f"Polynomial: {model.l1.data[0]} + {model.l1.data[1]}*x + {model.l1.data[2]}*x^2 + {model.l1.data[3]}*x^3 + {model.l1.data[4]}*x^4 + {model.l1.data[5]}*x^5")
print(f"Polynomial: {model.l1.data[0]} + {model.l1.data[1]}*x + {model.l1.data[2]}*x^2 + {model.l1.data[3]}*x^3 + {model.l1.data[4]}*x^4")

z = (XT.matmul(model.l1)).data

# plt.figure()
# plt.plot(xs,z)
# plt.xlabel('x') 
# plt.ylabel('y_estimate') 
# plt.title("Plot of y estimate")
# plt.show()


#################### test data ####################

with open('data_test.csv', newline='') as f:
    reader = csv.reader(f)
    test_data = list(reader)


op_test = np.array(test_data)

#remove column headers and convert string to float
op_test = op_test[1:].astype(float)

#shuffle training data for SGD
#np.random.shuffle(op)

#separate feature and label
x_test,y_test = op_test.T

#get sample size
m_test = np.shape(x_test)[0]

#convert 1D array to 2D array for operations
xs_test = np.reshape(x_test,(m_test,1))
ys_test = np.reshape(y_test,(m_test,1))

#plot
plt.figure()
plt.plot(xs_test,ys_test,"-b", label = "actual")
plt.xlabel('x') 
plt.ylabel('y') 


#feature scaling of the fifth order polynomial
x1_test = one2twoD(normalize(xs_test,x1_mean,x1_std)) 
x2_test = one2twoD(normalize(xs_test**2,x2_mean,x2_std))
x3_test = one2twoD(normalize(xs_test**3,x3_mean,x3_std))
x4_test = one2twoD(normalize(xs_test**4,x4_mean,x4_std))
# x5_test = one2twoD(normalize(xs_test**5))


X_test = Tensor(np.concatenate((np.ones((m_test,1)), x1_test, x2_test, x3_test, x4_test),axis = 1))

y_hat = (X_test.matmul(model.l1)).data

R = r_squared(ys_test, y_hat)

plt.plot(xs_test, y_hat,"-r", label = "predicted")
plt.legend(loc="upper center")
plt.title(f"Plot of test data with R-squared: {R}")
plt.show()





