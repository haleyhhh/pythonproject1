import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import EMNIST

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 47)  # 修改为47个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载EMNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_dataset = EMNIST(root='./data', split='balanced', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

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
def train_model(model, train_loader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# 模拟剪枝过程
def simulate_pruning(model, start_rate, end_rate, epochs):
    optimizer = model.optimizer  # 确保模型对象有一个属性指向其优化器
    for epoch in range(epochs):
        current_rate = start_rate + (end_rate - start_rate) * epoch / (epochs - 1)
        # 逐步剪枝：这里只是简单地减少学习率来模拟剪枝效果
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1 - current_rate)

# 训练和评估
epochs = 10
models, optimizers = initialize_models(1)  # 使用1个模型进行示例
criterion = nn.CrossEntropyLoss()

# 为模型设置属性以引用其优化器
for model, optimizer in zip(models, optimizers):
    model.optimizer = optimizer

# 训练 FedAvg
fedavg_model = models[0]
fedavg_accuracies = []
for epoch in range(epochs):
    train_model(fedavg_model, train_loader, fedavg_model.optimizer, criterion)
    accuracy = evaluate_model(fedavg_model, test_loader)
    fedavg_accuracies.append(accuracy)

# 模拟基于联邦剪枝的算法训练
pruned_model = models[0]
pruned_accuracies = []
simulate_pruning(pruned_model, 0.1, 0.0, epochs)  # 模拟剪枝过程，学习率从0.1降到0
for epoch in range(epochs):
    train_model(pruned_model, train_loader, pruned_model.optimizer, criterion)
    accuracy = evaluate_model(pruned_model, test_loader)
    pruned_accuracies.append(accuracy)

# 模拟Fedsel算法训练
fedsel_model = models[0]
fedsel_optimizer = optimizers[0]
fedsel_accuracies = []
num_samples = int(len(train_dataset) * 0.5)  # 选择50%的数据

def sample_dataset(train_dataset, num_samples):
    indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    sampled_dataset = torch.utils.data.Subset(train_dataset, indices)
    return sampled_dataset

fedsel_train_dataset = sample_dataset(train_dataset, num_samples)
fedsel_train_loader = DataLoader(fedsel_train_dataset, batch_size=64, shuffle=True)

for epoch in range(epochs):
    train_model(fedsel_model, fedsel_train_loader, fedsel_optimizer, criterion)
    accuracy = evaluate_model(fedsel_model, test_loader)
    fedsel_accuracies.append(accuracy)

# 绘制柱形图
epoch_indices = np.arange(1, epochs + 1)
plt.bar(epoch_indices - 0.4, fedavg_accuracies, 0.2, label='FedAvg', color='b')
plt.bar(epoch_indices - 0.2, pruned_accuracies, 0.2, label='Pruned', color='r')
plt.bar(epoch_indices + 0.2, fedsel_accuracies, 0.2, label='Fedsel', color='g')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Comparison of FedAvg, Pruned, and Fedsel Algorithms on EMNIST')
plt.xticks(epoch_indices)
plt.legend()
plt.show()