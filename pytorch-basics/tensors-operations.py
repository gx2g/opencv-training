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

"""
1. Converting Images to Batched tensors
An image is made up of pixel arrays that represent the intensity of pixels in grayscale or the color 
values in RGB format. When working with deep learning models, it's often necessary to convert these 
images into tensors, which are the primary data structures used in PyTorch for handling and 
processing data.

Tensors: In PyTorch, tensors are multi-dimensional arrays similar to NumPy arrays, but with additional 
capabilities for GPU acceleration and automatic differentiation. Tensors are the fundamental building 
blocks for representing data and parameters in neural networks.

Batches: Batching is a technique where multiple data samples (images, in this case) are grouped 
together into a single tensor. This allows efficient processing of multiple samples simultaneously, to 
take advantage of the parallel processing capabilities of modern hardware.
In the following block, we will see an example of converting two MNIST images into a single batched 
tensor of shape [2,3,28,28]
"""

# Download some digit images from MNIST dataset
digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

digit_0_array_gray = cv2.imread("mnist_0.jpg",cv2.IMREAD_GRAYSCALE )
digit_1_array_gray = cv2.imread("mnist_1.jpg",cv2.IMREAD_GRAYSCALE )

# Visualize the image



fig, axs = plt.subplots(1,2, figsize=(10,5))


axs[0].imshow(digit_0_array_og, cmap='gray',interpolation='none')
axs[0].set_title("Digit 0 Image")
axs[0].axis('off')

axs[1].imshow(digit_1_array_og, cmap="gray", interpolation = 'none')
axs[1].set_title("Digit 1 Image")
axs[1].axis('off')

plt.show()


# Numpy array with three channels
print("Image array shape: ", digit_0_array_og.shape)
print(f"Min pixel value: {np.min(digit_0_array_og)} ; Max pixel value : {np.max(digit_0_array_og)}")

# We will have a look at 28x28 single channel image's pixel values
print(digit_0_array_gray)




# 1.1. Convert Numpy array to Torch tensors 
# Convert the images to PyTorch tensors and normalize
img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) / 255.0
img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) / 255.0

print("Shape of Normalised Digit 0 Tensor: ", img_tensor_0.shape)
print(f"Normalised Min pixel value: {torch.min(img_tensor_0)} ; Normalised Max pixel value : {torch.max(img_tensor_0)}")

plt.imshow(img_tensor_0,cmap="gray")
plt.title("Normalised Digit 0 Image")
plt.axis('off')
plt.show()



# 1.2. Creating Input Batch
batch_tensor = torch.stack([img_tensor_0, img_tensor_1])

# In PyTorch the forward pass of input images to the model is expected to have a batch_size > 1
print("Batch Tensor Shape:", batch_tensor.shape)

"""
Additionally in PyTorch, image tensors typically follow the shape convention [N ,C ,H ,W] unlike 
tensorflow which follows [N, H, W, C].

Therefore, we need to bring the color channel to the second dimension. This can be achieved using 
either torch.view() or torch.permute().
"""

batch_input = batch_tensor.permute(0,3,1,2)

