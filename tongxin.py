import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # Adjusted for MNIST
        self.fc2 = nn.Linear(128, 10)  # MNIST has 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

batch_size = 64  # 您可以根据需要调整批大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型和优化器
def initialize_models(num_models):
    models = [SimpleCNN() for _ in range(num_models)]
    optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9) for model in models]
    return models, optimizers

# 评估模型性能
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total

# 训练模型
def train_model(model, train_loader, optimizer, criterion, dpba_epsilon=None):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if dpba_epsilon is not None:
            noise = torch.randn_like(loss) * (1 / dpba_epsilon)
            loss += noise
        loss.backward()
        optimizer.step()

# FedAvg算法
def fedavg_training(model, train_loader, optimizer, criterion, epochs):
    accuracy_history = []
    for epoch in range(epochs):
        train_model(model, train_loader, optimizer, criterion)
        accuracy = evaluate_model(model, test_loader)
        accuracy_history.append(accuracy)
    return accuracy_history

# Fedsel算法
def fedsel_training(model, train_loader, optimizer, criterion, epochs, selection_rate=0.5):
    accuracy_history = []
    for epoch in range(epochs):
        if np.random.rand() < selection_rate:
            train_model(model, train_loader, optimizer, criterion)
        accuracy = evaluate_model(model, test_loader)
        accuracy_history.append(accuracy)
    return accuracy_history

# 模型剪枝算法
def model_pruning(model, pruning_rate=0.1):
    for name, param in model.named_parameters():
        if 'conv' in name or 'fc' in name:  # 假设只对卷积层和全连接层进行剪枝
            num_params_to_prune = int(pruning_rate * param.numel())
            num_params_to_prune = max(0, min(num_params_to_prune, param.numel() - 1))
            if num_params_to_prune > 0:  # 确保有参数可以被剪枝
                values, indices = torch.topk(param.abs().view(-1), num_params_to_prune, largest=False)
                threshold = values[-1]  # 获取剪枝阈值
                mask = param.abs().ge(threshold).view_as(param)
                param.data.mul_(mask)

# 训练和评估
epochs = 10
models, optimizers = initialize_models(3)  # 使用3个模型进行示例
criterion = nn.CrossEntropyLoss()

fedavg_accuracies = fedavg_training(models[0], train_loader, optimizers[0], criterion, epochs)
fedsel_accuracies = fedsel_training(models[1], train_loader, optimizers[1], criterion, epochs, selection_rate=0.5)
model_pruning(models[2], pruning_rate=0.1)
pruned_accuracies = fedavg_training(models[2], train_loader, optimizers[2], criterion, epochs)

# 绘制结果
plt.plot(fedavg_accuracies, label='FedAvg')
plt.plot(pruned_accuracies, label='Pruned')
plt.plot(fedsel_accuracies, label='FedSel')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Comparison of FedAvg, Pruned, and FedSel Algorithms on MNIST')
plt.legend()
plt.show()