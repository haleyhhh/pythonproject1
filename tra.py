import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np  # 导入 NumPy 库

device = torch.device('cpu')

# 载入训练集和测试集
train_dataset = datasets.MNIST(root='./MNIST/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./MNIST/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# 设置批次大小
batch_size = 64

# 装载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(F.conv2d(x, self.conv1.weight, self.conv1.bias, padding=2))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, self.conv2.weight, self.conv2.bias, padding=2))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.linear(x, self.fc.weight, self.fc.bias)
        return F.log_softmax(x, dim=1)


# 初始化全局模型和本地模型
global_model = Net().to(device)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 初始化全局模型的优化器
global_optimizer = optim.SGD(global_model.parameters(), lr=0.01)


# 定义传统模型剪枝函数
def traditional_pruning(model, pruning_rate):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = torch.quantile(torch.abs(param.data), pruning_rate)
            mask = torch.abs(param.data) > threshold
            param.data[~mask] = 0  # 将不重要的权重置零

# 定义参与联邦学习的客户端数量
num_clients = 5
# 初始化本地模型列表
local_models = [Net().to(device) for _ in range(num_clients)]
# 创建客户端数据集的子集
client_datasets = [Subset(train_dataset, torch.arange(i * len(train_dataset) // num_clients, (i + 1) * len(train_dataset) // num_clients)) for i in range(num_clients)]

# 为每个客户端创建数据加载器
client_loaders = [DataLoader(dataset=client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]

# 定义本地模型的优化器列表
local_optimizers = [optim.SGD(local_model.parameters(), lr=0.01) for local_model in local_models]
# 定义联邦学习剪枝函数
def federated_pruning(global_model, local_models, pruning_rate):
    # 聚合模型
    global_model_state = global_model.state_dict()
    for key in global_model_state.keys():
        global_model_state[key] = torch.mean(
            torch.stack([local_model.state_dict()[key] for local_model in local_models]), dim=0)
    global_model.load_state_dict(global_model_state)

    # 剪枝
    for name, param in global_model.named_parameters():
        if 'weight' in name:
            threshold = torch.quantile(torch.abs(param.data), pruning_rate)
            mask = torch.abs(param.data) > threshold
            param.data[~mask] = 0

# 定义联邦学习训练函数
def federated_train(local_models, client_loaders, local_optimizers, criterion, num_epochs, client_sampling):
    for epoch in range(num_epochs):
        sampled_clients = np.random.choice(range(len(local_models)), size=client_sampling, replace=False)
        for client_idx in sampled_clients:
            local_model = local_models[client_idx]
            local_model.train()
            for data in client_loaders[client_idx]:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                local_optimizers[client_idx].zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                local_optimizers[client_idx].step()
    return local_models
# 定义聚合模型参数的函数
def aggregate_models(models):
    aggregated_model_state = {}
    for model in models:
        for name, param in model.named_parameters():
            if name not in aggregated_model_state:
                aggregated_model_state[name] = param.data.clone()
            else:
                aggregated_model_state[name] += param.data
    for name, param in aggregated_model_state.items():
        aggregated_model_state[name] /= len(models)
    return aggregated_model_state

# 联邦学习训练
federated_train(local_models, client_loaders, local_optimizers, criterion, 20, num_clients // 2)


# 聚合模型
global_model_state = aggregate_models(local_models)
global_model.load_state_dict(global_model_state)


# 定义训练模型的函数
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
    return epoch_losses, epoch_accuracies


# 定义测试模型的函数
def test_model(model, test_loader, criterion):
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss /= len(test_loader)
    test_accuracy = correct / total
    return test_loss, test_accuracy

# 传统模型剪枝
traditional_pruning(global_model, 0.5)
# 联邦学习剪枝
federated_pruning(global_model, local_models, 0.3)

# 训练剪枝后的模型
train_losses_traditional, train_accuracies_traditional = train_model(global_model, train_loader, global_optimizer, criterion, 20)
# 测试剪枝后的模型
test_loss_traditional, test_accuracy_traditional = test_model(global_model, test_loader, criterion)

# 联邦学习剪枝
federated_pruning(global_model, local_models, 0.3)

# 训练剪枝后的全局模型
train_losses_federated, train_accuracies_federated = train_model(global_model, train_loader, global_optimizer, criterion, 20)
# 测试剪枝后的全局模型
test_loss_federated, test_accuracy_federated = test_model(global_model, test_loader, criterion)
# 更新全局模型
global_model.load_state_dict(global_model_state)

# 绘制传统模型剪枝和联邦学习剪枝的对比图
plt.figure()
plt.plot(range(1, 21), train_losses_traditional, label='Train Loss Traditional')
plt.plot(range(1, 21), train_losses_federated, label='Train Loss Federated')
plt.title('Train Loss Comparison Between Traditional and Federated Pruning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, 21), train_accuracies_traditional, label='Train Accuracy Traditional')
plt.plot(range(1, 21), train_accuracies_federated, label='Train Accuracy Federated')
plt.title('Train Accuracy Comparison Between Traditional and Federated Pruning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 单独绘制测试损失和准确率
plt.figure()
plt.plot(1, test_loss_traditional, label='Test Loss Traditional')
plt.plot(1, test_loss_federated, label='Test Loss Federated')
plt.title('Test Loss Comparison Between Traditional and Federated Pruning')
plt.xlabel('Test')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(1, test_accuracy_traditional, label='Test Accuracy Traditional')
plt.plot(1, test_accuracy_federated, label='Test Accuracy Federated')
plt.title('Test Accuracy Comparison Between Traditional and Federated Pruning')
plt.xlabel('Test')
plt.ylabel('Accuracy')
plt.legend()
plt.show()