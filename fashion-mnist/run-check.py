# -*- coding: utf-8 -*-
"""
@authors: Bo Deng, bdeng@unl.edu, UNL
          Levi Heath, lheath2@unl.edu, UNL
@license: CC BY-NC-ND 4.0
"""

# Check Code for fashion_mnist Error-Free ANN Model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


# Parameters for 784-n-10 ANN architecture
input_size = 784 # size of input vectors which are images of integers 0 to 9
nmb_of_h_nodes = 100 # number of hidden layers
nmb_of_labels = 10 # input data is images of integers 0 to 9 


# Load fashion_mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
y_train = to_categorical(y_train, nmb_of_labels).astype('float32')
y_test = to_categorical(y_test, nmb_of_labels).astype('float32')
nmb_of_samples =  int(tf.shape(x_train)[0])


# load model
model = tf.keras.models.load_model('FashionMNIST_ErrorFree_model.keras')


""" 
    Functions for Computing Positive Rate (PR)/Catagorical Accuracy
"""
# function for determining predicted labels
def one_hot_max_per_row(tensor):
    # Find the index of the maximum value in each row
    max_indices = tf.argmax(tensor, axis=1)
    
    # Create a one-hot encoded tensor with a 1 at the position of the maximum value
    one_hot_tensor = tf.one_hot(max_indices, depth=tensor.shape[1])
    
    return one_hot_tensor


# function to determine the indices of the incorrectly predicted labels
def find_row_indices(tensor1, tensor2, string):
    # Ensure the tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("The input tensors must have the same shape for the function find_unequal_row_indices.")
    if string not in ['equal','unequal']:
        raise ValueError("Third input must be the string 'equal' or 'unequal'.")
    
    # Find the row indices where the tensors are unequal
    unequal_indices = tf.where(tf.reduce_any(tf.math.not_equal(tensor1, tensor2), axis=1))
    # Convert the tensor of indices to a NumPy array
    unequal_indices_np = unequal_indices.numpy().reshape(1,-1)[0]
    
    # Determine equal row indices
    indices = np.arange(tensor1.shape[0])
    mask = ~np.isin(indices,unequal_indices_np)
    equal_indices_np = indices[mask]
    
    if string == 'unequal':
        return unequal_indices_np
    if string == 'equal':
        return equal_indices_np

# Function for computing PR
def cat_acc(target_prediction,current_prediction):
    target = one_hot_max_per_row(target_prediction)
    current = one_hot_max_per_row(current_prediction)
    
    number_incorrect = np.shape(find_row_indices(target, current,'unequal'))[0]
    total = target_prediction.shape[0]
    
    PR = (total - number_incorrect)/total
    
    return PR


""" 
    Compute Positive Rate (PR)
"""
y_train_pred = model(x_train)
PR_train = cat_acc(y_train, y_train_pred)

y_test_pred = model(x_test)
PR_test = cat_acc(y_test, y_test_pred)


print('Positive rate on training set: ', PR_train, '\n Positive rate on test set: ', PR_test)
