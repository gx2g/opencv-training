# Use this if you have conda installed
# !conda install -c pytorch pytorch

# Use this if you are on Google Colab
# or don't have conda installed
# !pip3 install torch


import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("torch version : {}" .format(torch.__version__))

# 2. Introduction to Tensors and its Operations

"""
We have seen the importance of tensors, now will understand it from ground up. Tensor is simply a 
fancy name given to matrices. If you are familiar with NumPy arrays, understanding and using 
PyTorch Tensors will be very easy. A scalar value is represented by a 0-dimensional Tensor. 
Similarly, a column/row matrix is represented using a 1-D Tensor and so on. Some examples of 
Tensors with different dimensions are shown for you to visualize and understand.
"""
# Explain dimensions, custom value assignment

# 0 dimensions
A = torch.tensor(3)
print("0 dimensions", A)

# 1 dimension
B = torch.tensor([1.0, 2.0, 3.0])
print("1 dimension", B)

# 2 dimensions
C = torch.tensor([
    [1.0, 2.0],[3.0, 4.0]
    ])
print("2 dimensions", C)

# 3 dimensions
D = torch.tensor([
    [[1.0, 2.0],[3.0, 4.0]], 
    [[5.0, 6.0],[7.0, 8.0]]
    ])
print("3 dimensions", D)

print("========================================")
# 2.1. Construct your first Tensor

# Create a Tensor with just ones in the column
a = torch.ones(5)
print("all ones", a)

b = torch.zeros(5)
print("all zeros", b)

c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("custom values", c)

d = torch.zeros(3,2)
print("3 rows, 2 columns of zeros", d)

e = torch.ones(3,2)
print("3 rows, 2 columns of ones", e)


f = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("2 demensions again", f)

# 3D
g = torch.tensor([
    [[1., 2.], [3., 4.]], 
    [[5., 6.], [7., 8.]]
    ])
print("3 demensions again", g)

# shape method
print(f.shape)
print(e.shape)
print(g.shape)