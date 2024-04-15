import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

# 定义网络结构
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

# 准备数据
def make_train_data_and_model():

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化网络和优化器
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return train_loader,optimizer,criterion,model

def make_test_data_and_model():

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 初始化网络和优化器
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return test_loader,optimizer,criterion,model


def train_and_save_weight():
    epochs = 1
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
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}')
    torch.save(model.state_dict(), 'model_weights.pth')


def eval_model(weights):

    test_loader,optimizer,criterion,model = make_test_data_and_model()
    # 加载权重
    model.load_state_dict(weights)

    # 将模型设置为评估模式
    model.eval()

    for images, labels in test_loader:
            output = model(images)
            loss = criterion(output, labels)
    
    return loss

# 修改权重
def make_vectors(path):
    weights_path = path
    weights = torch.load(weights_path)


    random_tensor_1 = {}
    random_tensor_2 = {}

    for name, param in weights.items():
        # 使用torch.randn_like来生成匹配的随机Tensor
        matched_tensor_1 = torch.randn_like(param) * 0.001
        random_tensor_1[name] = matched_tensor_1
        
        matched_tensor_2 = torch.randn_like(param) * 0.001
        random_tensor_2[name] = matched_tensor_2

    return random_tensor_1,random_tensor_2,weights

def modify_weights(path):
    v1,v2, weights =  make_vectors(path)

    step = 6

    map = []
    weights_now= weights

    for i in range(step):
        for j in range(step):
            for name, tensor in weights.items():

                ii = i - step/2
                jj = j - step/2
                weighting_change = v1[name]*(ii)  + v2[name]*(jj)
                weights_now[name] = weights[name] + weighting_change
            loss = float(eval_model(weights_now))

            map.append([ii,jj,loss])
    with open('map_data.pkl', 'wb') as f:
        pickle.dump(map, f)

modify_weights('plot_loss_landscape/model_weights.pth')