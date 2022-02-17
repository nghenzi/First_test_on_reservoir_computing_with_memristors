# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:48:26 2022

@author: 54911
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob 
from collections import Counter 

filt = lambda x: '_' in x
get_letter = lambda x: x.split('.')[0]
# for i in range(10):
letters = list(filter(filt,glob('*.npy')))

letters= ['l0_ya.npy',  'l1_yu.npy',  'l2_oi.npy', 'l3_yoi.npy', 'l4_yai.npy', 'l5_p.npy',
 'l6_m.npy', 'l7_t.npy', 'l8_r.npy', 'l9_b.npy', 
 'letsbuy.npy', 'letsgo.npy', 'letsride.npy', 'kick.npy','getout.npy']

letters = letters[:10]
ls = list(map(np.load, letters)) # letters[:10], letters[10:]
d = dict(zip(list(map(get_letter,letters)),ls))
d # in this dictionary is the data of the different letters to recognize 
#%% plot of the letters to recognize 
letters = [ 'l0_ya.npy', 'l1_yu.npy', 'l2_oi.npy', 'l3_yoi.npy', 'l4_yai.npy',
 'l5_p.npy', 'l6_m.npy', 'l7_t.npy', 'l8_r.npy', 'l9_b.npy']
fig, ax = plt.subplots(len(letters)//2,2)
ax = [a for ae in ax for a in ae]
for i, lett in enumerate(letters):
    ax[i].imshow(np.load(lett), cmap=plt.cm.Greens, clim=[-1,2])
    ax[i].axis('off') 
    
plt.subplots_adjust(wspace=0.05,hspace=0.05)

#%%
fig, ax = plt.subplots(2,1)

t = np.arange(6)
# plt.plot(t,  np.cumsum(1/150*np.exp(-t)),'-o')

inp =  np.array([1., 0., 1., 0., 1.])

def output_row(initial_state, input_signal): 
    """
    This function emulates the behavior of a memristor PoC.
    ㅑnsert here a more detailed model or integrate a differential equation 
    for better behavior.
    Ideally the experimental data should be here. 
    
    Parameters
    ----------
    initial_state : conductance value before any pulse.
    input_signal : signal of the applied pulses.
    Returns
    -------
    a : the output of the conductance. this has to be extracted of the device. 
    """
    a = [initial_state]

    for i in range(5):
        if input_signal[i] > 0 :
            a.append(np.clip(a[i],0.1,1)*np.exp(1))
        else:
            a.append(np.clip(a[i],1,10)*(3-np.exp(1)))
    return a

matrix = np.zeros((10,31))
nr=0
for nl, lett in enumerate(d.keys()):
    print (nl,nr) 
    for nr, row in enumerate(d[lett]):
        initial_state = np.random.random(1)
        output = output_row(initial_state,row)
        ax[0].plot(output+np.random.random(1)*1e-4,'-o')
        matrix[nl,nr*6:(nr+1)*6] = output 

matrix[:,30] = 2.5* np.random.random((10,))
         
# plt.figure()
ax[1].imshow(matrix, extent= [10,1,1,31],aspect='auto')   

# plt.plot(a)

#%%
"""
one hot encoding, softmax function activation and training procedure  
"""

def one_hot(y, c):    
    """
    # y--> label/ground truth.
    # c--> Number of classes.
    """ 

    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))
    
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    
    return y_hot


one_hot([5,2,3], 10)

def softmax(z):    
    # z--> linear part.
    
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))
    
    # Calculating softmax for all example letters.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp

# fit(X,np.arange(10),0.1,10,100)
def fit(X, y, lr, c, epochs):
    """
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.
    """
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)
    # Empty list to store losses.
    losses = []
    
    np.save('initial_w.npy',w)
    # Training loop.
    for epoch in range(epochs):
        
        # Calculating hypothesis/prediction.
        z = X@w + b
        y_hat = softmax(z)
        
        # One-hot encoding y.
        y_hot = one_hot(y, c)
        
        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
        b_grad = (1/m)*np.sum(y_hat - y_hot)
        
        # Updating the parameters.
        w = w - lr*w_grad
        b = b - lr*b_grad
        
        np.save('w' + str(epoch) + '.npy',w)
        # Calculating loss and appending it in the list.
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)
        # Printing out the loss at every 100th iteration.
        if epoch%1==0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))
    return w, b, losses

# w, b, losses = fit(X,np.arange(10),0.1,10,100)
#%% training 
"""  10 examples for the training, each example has 6*5 features, or 30 virtual reservoir nodes 
"""

X = np.zeros((10,30))
for i, letter in enumerate(d.keys()):
    initial_state = np.random.random(1)
    output = []
    for row in d[letter]:
        output.append(output_row(initial_state,row))
    X[i,:] = np.concatenate(output)[:,0]


w, b, losses = fit(X,np.arange(10),0.1,10,100)

plt.figure()
plt.plot(losses) 

#%% testing 
"""
here I add noise in each position of the 25 positions of the matrix
"""
cc = ['l0_ya.npy', 'l1_yu.npy', 'l2_oi.npy', 'l3_yoi.npy',
 'l4_yai.npy',  'l5_p.npy', 'l6_m.npy', 'l7_t.npy',
 'l8_r.npy',  'l9_b.npy']

# fig345, ax345 = plt.subplots(5,5)
# test_letter  = d[cc[0].split('.')[0]]
# for i in range(5):
#     for j in range(5):
#         test_letter  = d[cc[0].split('.')[0]].copy()
#         test_letter[i,j] = 1 if not test_letter[i,j] else 0
#         ax345[i,j].imshow(np.array(test_letter.copy()), cmap='jet')

# for i in range(5):
#     for j in range(5):
#         print (5*i+j)

"""  10 examples for the training, each example has 6*5 features, or 30 virtual reservoir nodes 
"""
X = np.zeros((25,30))


num_letter =  0# between 0 and 9 
# test_letter  = d[cc[0].split('.')[0]]
fig345, ax345 = plt.subplots(5,5)
for i in range(5):
    for j in range(5):
        test_letter  = d[cc[num_letter].split('.')[0]].copy()
        test_letter[i,j] = 1 if not test_letter[i,j] else 0
        initial_state = np.random.random(1)
        output = []
        case_letter = np.array(test_letter.copy())
        ax345[i,j].imshow(np.array(case_letter),cmap=plt.cm.Greens, clim=[-1,2])
        for row in case_letter:
            output.append(output_row(initial_state,row))
        X[5*i+j,:] = np.concatenate(output)[:,0]
             

def predict(X, w, b):
    """ X --> Input.
        w --> weights.
        b --> bias."""
    
    # Predicting
    z = X@w + b
    y_hat = softmax(z)
    # print (y_hat.shape)
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)
    # return y_hat
    
# for i in range(30)
print ('prediction of the corresponding letter', num_letter, ':', predict(X, w, b))


#%% testing 

def predict(X, w, b):
    """ X --> Input.
        w --> weights.
        b --> bias."""
    
    # Predicting
    z = X@w + b
    y_hat = softmax(z)
    # print (y_hat.shape)
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

"""  test the 10 cases for making the confusion matrix 
"""
X = np.zeros((25,30))

confusion_matrix = np.zeros((10,10))

for num_letter in range(10):# between 0 and 9 
    # test_letter  = d[cc[0].split('.')[0]]
    fig345, ax345 = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            test_letter  = d[cc[num_letter].split('.')[0]].copy()
            test_letter[i,j] = 1 if not test_letter[i,j] else 0
            initial_state = np.random.random(1)
            output = []
            case_letter = np.array(test_letter.copy())
            ax345[i,j].imshow(np.array(case_letter),cmap=plt.cm.Greens, clim=[-1,2])
            for row in case_letter:
                output.append(output_row(initial_state,row))
            X[5*i+j,:] = np.concatenate(output)[:,0]
                 
    predictions = predict(X, w, b)
    print ('prediction of the corresponding letter', num_letter, ':', predictions)
    for n_lett, prob in Counter(predictions).items():
        confusion_matrix[num_letter, n_lett] = prob/25*100.
    
plt.figure()
plt.imshow(confusion_matrix) 

    
#%%
# plt.figure()
# plt.imshow(X)

#%%  video of the training process 

import numpy as np, matplotlib.pyplot as plt
import threading as th, time

def update(fig, ax, im):
    for frame in range(100): 
        # aux = np.roll(im.get_array(), -1, axis=0)
        # aux[-1,:] = np.arange(14260)*0        
        im.set_data( np.load('w' + str(frame) + '.npy'))
        fig.canvas.draw()
        print (frame)
        time.sleep(0.1)
            
def main():
    fig, ax = plt.subplots()
    im = ax.imshow(np.load('initial_w.npy'), 
                   aspect='auto', clim=[0.5,1])
    th_plot = th.Thread(target=update, args=[fig, ax, im])
    th_plot.setDaemon(True)
    th_plot.start()
    plt.show()
    
main() 

