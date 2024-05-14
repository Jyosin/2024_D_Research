import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import pickle

# 定义修改和评估函数

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
print(f'Using device: {device}')


def make_test_data_and_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以适应 ViT 模型输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    # 初始化 Vision Transformer 网络和优化器
    model = models.vit_b_16(weights=None, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return test_loader, optimizer, criterion, model

def eval_model(weights):
    test_loader, optimizer, criterion, model = make_test_data_and_model()
    model.load_state_dict(weights)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(test_loader.dataset)
    return average_loss


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
    step = 50
    map = []

    for i in range(step):
        for j in range(step):
            weights_now = {name: tensor.clone().to(device) for name, tensor in original_weights.items()}
            ii = i - (step // 2)
            jj = j - (step // 2)
            for name in original_weights:
                weighting_change = v1[name].to(device) * ii * 0.1 + v2[name].to(device) * jj * 0.1
                weights_now[name] = weights_now[name] + weighting_change
            
            loss = eval_model(weights_now)
            map.append([ii, jj, loss])
            print(i * step + j)
    
    with open('result/vit_untrained_0.1_50.pkl', 'wb') as f:
        pickle.dump(map, f)

# 此处需更改路径至适当的模型权重文件
modify_weights('vit_untrained.pth')
