import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Download some digit images from MNIST dataset
import os
import requests

urls = {
    "mnist_0.jpg": "https://learnopencv.com/wp-content/uploads/2024/07/mnist_0.jpg",
    "mnist_1.jpg": "https://learnopencv.com/wp-content/uploads/2024/07/mnist_1.jpg"
}

for filename, url in urls.items():
    if not os.path.exists(filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename} from {url}")
    else:
        print(f"File {filename} already exists. Skipping download.")

# print version of pytorch
print("torch version : {}".format(torch.__version__))



# Create a 100x200 image with 3 color channels (RGB), initialized to zeros (black image)
image = np.zeros((100, 200, 3), dtype=np.uint8)

# Let's set the left half to orange and the right half to blue as an example:
image[:, :100] = [255, 128, 0]   # Orange (R=255, G=128, B=0)
image[:, 100:] = [0, 0, 255]     # Blue   (R=0, G=0, B=255)

print(image.shape)  # Output: (100, 200, 3)

# show image color in plot
plt.imshow(image)
plt.axis('off')  # Optional: hides the axis
plt.show()


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

#Numpy array with three channels
print("Image array shape: ",digit_0_array_og.shape)

# np.min and mp.max pixel values
print(f"Min pixel value: {np.min(digit_0_array_og)} ; Max pixel value: {np.max(digit_0_array_og)}")
      
# We will have a look at 28x28 single channel image's pixel values
digit_0_array_gray

# print the type of data
print(type(digit_0_array_gray))


# 1.1. Convert Numpy array to Torch tensors

# Convert the original image arrays (digit_0_array_og and digit_1_array_og) 
# into PyTorch tensors and normalizes the pixel values.

# creates a PyTorch tensor from the digit_0_array_og NumPy array
img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) / 255.0 
# dtype=torch.float32 specifies that the elements of the tensor should be 32-bit floating-point 
# numbers. This is a common data type for numerical computations in deep learning

img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) / 255.0
# / 255.0 normalizes the pixel values by dividing each pixel value by 255.0. 
# Image pixel values typically range from 0 to 255. 
# Normalizing them to the range [0, 1] is a standard practice in image processing 
# for neural networks, as it can help improve training stability and performance.


# This line prints the shape of the resulting img_tensor_0. The shape will be (28, 28, 3), representing 
# the height, width, and number of color channels (RGB) of the image.
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


# Additionally in PyTorch, image tensors typically follow the shape 
# convention [N ,C ,H ,W] unlike tensorflow which follows [N, H, W, C].
# Therefore, we need to bring the color channel to the second dimension. 
# This can be achieved using either torch.view() or torch.permute().

batch_input = batch_tensor.permute(0,3,1,2)
print("Batch Tensor Shape:", batch_input.shape)


# 2. Introduction to Tensors and its Operations

# Read the image from local file (replace 'my_image.jpg' with your filename)
img = cv2.imread('pytorch-tensors-example.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or unable to read.")
else:
    # Show the image in a window
    cv2.imshow('Understanding Tensors - Examples', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the window

a = torch.tensor(3)
print('example of 0 dimensons:', a)
b = torch.tensor([1.0,2.0,3.0])
print('example of 1 dimensons:', b)
c = torch.tensor([[1.0,2.0],[3.0,4.0]])
print('example of 2 dimensons:', c)
d = torch.tensor([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])
print('example of 3 dimensons:', d)

if d.ndim == 3:
    print(type(d), "variable array: is 3 dimensions")
else:
    print("Array is not 3D")
