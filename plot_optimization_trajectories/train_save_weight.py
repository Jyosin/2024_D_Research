import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from torchvision.datasets import CIFAR10

# 定义网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 640)
        self.fc3 = nn.Linear(640, 1280)
        self.fc4 = nn.Linear(1280, 320)
        self.fc5 = nn.Linear(320, 64)  # 隐藏层
        self.fc6 = nn.Linear(64, 10)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 准备数据
def make_train_data_and_model():


    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载CIFAR-10数据集
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

    # 初始化网络和优化器
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return train_loader,optimizer,criterion,model


def train_and_save_weight():
    epochs = 50
    train_loader,optimizer,criterion,model = make_train_data_and_model()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(model.state_dict(), f'/home/wang/2024_D_Research/data/CIFA_10/without_modified/train/model_weights_5layers_epoch_{epoch + 1}.pth')
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}')
    torch.save(model.state_dict(), '/home/wang/2024_D_Research/data/CIFA_10/without_modified/train/model_weights_e.pth')



    print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}')   

train_and_save_weight()