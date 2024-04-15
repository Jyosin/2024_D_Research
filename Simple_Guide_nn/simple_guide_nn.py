import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义SimpleNN
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 10)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义GuideNN
class GuideNN(nn.Module):
    def __init__(self, input_size):
        super(GuideNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 假设输出是对SimpleNN学习率的建议调整

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 准备数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化网络和优化器
model = SimpleNN()
initial_lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 为了确定GuideNN的输入大小，我们先做一个假的前向传播以获得权重向量的大小
dummy_input = torch.zeros(1, 28*28)
dummy_output = model(dummy_input)
weight_vector_size = sum(p.numel() for p in model.parameters()) + 1  # 加1是为了包括loss
guide_model = GuideNN(weight_vector_size)
guide_optimizer = optim.SGD(guide_model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.view(images.size(0), -1), labels
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 创建权重和loss的向量
    with torch.no_grad():  # 确保不会计算这部分的梯度
        weight_loss_vector = torch.cat([p.data.view(-1) for p in model.parameters()] + [torch.tensor([total_loss])])
        guide_signal = guide_model(weight_loss_vector.unsqueeze(0))  # 增加batch维度

    # 使用GuideNN的输出调整SimpleNN的学习率
    lr_adjustment = guide_signal.item()
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr * (1 + lr_adjustment)

    print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}, LR Adjustment: {lr_adjustment}')

target_adjustment = torch.tensor([0.0])

for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.view(images.size(0), -1), labels
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    with torch.no_grad():
        # 创建权重和loss的向量
        weight_loss_vector = torch.cat([p.data.view(-1) for p in model.parameters()] + [torch.tensor([total_loss])])
    
    # 计算GuideNN的输出（学习率调整信号）
    guide_signal = guide_model(weight_loss_vector.unsqueeze(0))

    # 训练GuideNN
    guide_optimizer.zero_grad()
    guide_loss = (guide_signal - target_adjustment).pow(2)  # 简单的损失函数：尽量让输出接近目标调整因子
    guide_loss.backward()
    guide_optimizer.step()

    # 使用GuideNN的输出调整SimpleNN的学习率
    lr_adjustment = guide_signal.item()
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr * (1 + lr_adjustment)

    print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}, LR Adjustment: {lr_adjustment}, Guide Loss: {guide_loss.item()}')
# 注意：这个简化的示例不包括GuideNN的训练过程


