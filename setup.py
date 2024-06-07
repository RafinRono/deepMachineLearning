import torch
import intel_extension_for_pytorch as ipex
from torch import nn
from PIL import Image

print(torch.__version__)
print(torch.cuda.is_available())

tensor = torch.tensor([1,2,3])
print(tensor, tensor.device)

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib as plt

print(torch.__version__, torchvision.__version__)

#image = torchvision.io.read_img('./welcome.jpg')

#transform = transforms.Compose([transforms.ToTensor()])

#image = transform(image)

#import matplotlib.pyplot as plt
#print(f"Image shape: {image.shape}")
#plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
#plt.show()

image = Image.open('images/clock.png')

# Define a transform to convert PIL
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = transform(image)

# print the converted Torch tensor
print(img_tensor)

import matplotlib.pyplot as plt
print(f"Image shape: {img_tensor.shape}")
plt.imshow(img_tensor.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
print(f"Image shape after squeeze: {img_tensor.squeeze().shape}")
plt.show()

print("end")