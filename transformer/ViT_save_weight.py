import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

print("0513")

device  = torch.device("cuda" if torch.cuda.is_available()else "cpu")
print(f'Using device : {device}')
# Step 1: 加载 Vision Transformer 模型

model = models.vit_b_16(weights=None, num_classes=100).to(device)
model = nn.DataParallel(model)  # 用于 CIFAR-100，有100个类别

# Step 2: 下载 CIFAR-100 数据集并准备 DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小以适应 ViT 模型输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True,num_workers=1,pin_memory=True,persistent_workers=True)

test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=1,pin_memory=True,persistent_workers=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#
#  Step 3: 保存模型权重
best_loss = float('inf')
early_stopping_patience = 100
no_improvement_counter = 0

for epoch in range(1000):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/100")
    with tqdm(total=len(train_loader), desc = f"Training Epoch {epoch+1}") as pbar:
        for images,labels in train_loader:
            try:
                images, labels = images.to(device),labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss':loss.item()})
                pbar.update(1)
            except Exception as e:
                print(f"Error loading batch: {e}")

    avg_loss = running_loss/len(train_loader)
    print(f'Epoch [{epoch+1}/100], Loss:{avg_loss:.4f}')

    torch.cuda.empty_cache()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images,labels in test_loader:
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)


            val_loss += loss.item()

    avg_val_loss = val_loss/len(test_loader)
    print(f'Validation Loss:{avg_val_loss:.4f}')



    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improvement_counter = 0
        torch.save(model.state_dict(), 'best_vit_weights.pth')

    else:
        no_improvement_counter += 1

    if no_improvement_counter >= early_stopping_patience:
        print(f"Early stopping after{epoch + 1} epochs. ")
        break

    scheduler.step()

    torch.cuda.empty_cache()

print("train is finished")


