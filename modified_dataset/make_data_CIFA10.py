import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

# Set random seed for reproducibility
torch.manual_seed(0)

# Define the transform with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Load CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(cifar10_train, batch_size=len(cifar10_train), shuffle=True)
data_iter = iter(train_loader)

# Get the images and labels
images, labels = next(data_iter)

# Print the original distribution
print("Original distribution:", Counter(labels.numpy()))

# Define the target distribution for the imbalanced dataset
target_distribution = {0: 500, 1: 5000, 2: 4500, 3: 4000, 4: 3500, 5: 3000, 6: 2500, 7: 2000, 8: 1500, 9: 1000}

# Create a new training dataset with the target distribution
indices = []
for class_index, count in target_distribution.items():
    class_indices = (labels == class_index).nonzero(as_tuple=True)[0]
    selected_indices = np.random.choice(class_indices, count, replace=True)
    indices.extend(selected_indices)

# Combine all selected indices to create the new dataset
new_indices = np.array(indices)
new_images = images[new_indices]
new_labels = labels[new_indices]

# Print the new distribution
print("New distribution:", Counter(new_labels.numpy()))

# Save the new dataset
torch.save(new_images, '/home/wang/2024_D_Research/data/CIFA_10/modified/modified_cifar_images.pt')
torch.save(new_labels, '/home/wang/2024_D_Research/data/CIFA_10/modified/modified_cifar_labels.pt')
