import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from torchvision.datasets import CIFAR10
# from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# 定义网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 10)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_test_data_and_model():

    loaded_images = torch.load('/home/wang/2024_D_Research/data/CIFA_10/modified/modified_cifar_images.pt')
    loaded_labels = torch.load('/home/wang/2024_D_Research/data/CIFA_10/modified/modified_cifar_labels.pt')

    # 重新创建TensorDataset
    reconstructed_dataset = TensorDataset(loaded_images, loaded_labels)
    test_loader = DataLoader(reconstructed_dataset, batch_size=1, shuffle=True)

    # 初始化网络和优化器
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return test_loader,optimizer,criterion,model



def eval_model(weights):
    test_loader, optimizer, criterion, model = make_test_data_and_model()
    model.load_state_dict(weights)
    model.eval()

    total_loss = 0
    with torch.no_grad():  # 不计算梯度，以加速和节省内存
        for images, labels in test_loader:
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)  # 计算这一批的总损失

    average_loss = total_loss / len(test_loader.dataset)  # 计算平均损失
    return average_loss

# 修改权重
def make_vectors(path):
    weights_path = path
    weights = torch.load(weights_path)


    random_tensor_1 = {}
    random_tensor_2 = {}

    for name, param in weights.items():
        # 使用torch.randn_like来生成匹配的随机Tensor
        matched_tensor_1 = torch.randn_like(param) * 1
        random_tensor_1[name] = matched_tensor_1
        
        matched_tensor_2 = torch.randn_like(param) * 1
        random_tensor_2[name] = matched_tensor_2

    return random_tensor_1,random_tensor_2,weights

def modify_weights(path):
    v1, v2, original_weights = make_vectors(path)

    step = 20
    map = []

    for i in range(step):
        for j in range(step):
            # 在每次迭代中使用权重的深拷贝
            weights_now = {name: tensor.clone() for name, tensor in original_weights.items()}
            
            # 计算新的权重偏移
            ii = i - (step // 2)
            jj = j - (step // 2)
            for name in original_weights:
                weighting_change = v1[name] * ii * 0.1 + v2[name] * jj* 0.1
                weights_now[name] = weights_now[name] + weighting_change
            
            # 评估当前权重配置的模型
            loss = eval_model(weights_now)
            map.append([ii, jj, loss])
            print(i * step + j)

    # 保存结果
    with open('/home/wang/2024_D_Research/data/CIFA_10/modified/map_epoch_200_step_20_0.1.pkl', 'wb') as f:
        pickle.dump(map, f)

modify_weights('/home/wang/2024_D_Research/data/CIFA_10/modified/train/model_weights_epoch_200.pth')