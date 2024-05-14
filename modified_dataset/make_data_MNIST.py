import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter

# 设置随机种子以确保结果的可重复性
torch.manual_seed(0)

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=len(mnist_train), shuffle=True)
data_iter = iter(train_loader)

# 使用 next() 函数而不是 .next()
images, labels = next(data_iter)

# 查看原始数据分布
print("Original distribution:", Counter(labels.numpy()))

# 设定目标分布
target_distribution = {0: 100, 1: 6000, 2: 5000, 3: 4000, 4: 3000, 5: 2000, 6: 1500, 7: 1200, 8: 800, 9: 500}

# 创建新的训练数据集
indices = []
for digit, count in target_distribution.items():
    # 找到该数字所有图像的索引
    digit_indices = (labels == digit).nonzero(as_tuple=True)[0]
    # 如果指定数量大于实际可用数量，允许重复抽样
    selected_indices = np.random.choice(digit_indices, count, replace=True)
    indices.append(selected_indices)

# 合并所有选择的索引并创建新的数据集
new_indices = np.concatenate(indices)
new_images = images[new_indices]
new_labels = labels[new_indices]

# 查看新数据集的分布
print("New distribution:", Counter(new_labels.numpy()))

# 保存调整后的数据集
torch.save(new_images, 'modified_mnist_images.pt')
torch.save(new_labels, 'modified_mnist_labels.pt')
