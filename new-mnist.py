import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 定义一个简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载CIFAR10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 基于数据特征的自适应隐私预算分配
def adaptive_epsilon(data_loader):
    total_variance = 0
    num_samples = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1)
            total_variance += data.var(dim=0).sum().item()
            num_samples += data.size(0)
    average_variance = total_variance / num_samples
    return 1.0 / (average_variance + 1e-6)

# 基于模型相关性的隐私预算分配
def model_related_epsilon(model):
    return 1.0 / (len(list(model.parameters())) + 1e-6)

# 训练和评估模型
def train_and_evaluate(model, train_loader, test_loader, optimizer, epsilon, epochs=10):
    model.train()
    total_loss = 0
    total_correct = 0
    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
            if epsilon > 0:
                # 添加噪声
                noise = torch.randn_like(data) * epsilon
                data += noise
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
    accuracy = total_correct / len(test_loader.dataset)
    return total_loss / len(train_loader), accuracy

# FedAvg算法实现
def fedavg_train(model, train_loader, test_loader, optimizer, epochs=10):
    model.train()
    total_loss = 0
    total_correct = 0
    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
    accuracy = total_correct / len(test_loader.dataset)
    return total_loss / len(train_loader), accuracy

# 主函数
def main():
    epochs = 10
    epsilon_values = np.linspace(0, 1, 11)  # 生成从0到1的11个epsilon值
    accuracies_no_dp = []
    losses_no_dp = []
    accuracies_data_dependent = []
    losses_data_dependent = []
    accuracies_model_related = []
    losses_model_related = []
    accuracies_combined = []
    losses_combined = []
    accuracies_fedavg = []
    losses_fedavg = []

    # 不使用隐私预算分配
    model_no_dp = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer_no_dp = optim.SGD(model_no_dp.parameters(), lr=0.01, momentum=0.9)
    for epsilon in epsilon_values:
        loss_no_dp, accuracy_no_dp = train_and_evaluate(model_no_dp, train_loader, test_loader, optimizer_no_dp, epsilon)
        accuracies_no_dp.append(accuracy_no_dp)
        losses_no_dp.append(loss_no_dp)

    # 基于数据特征的自适应隐私预算分配
    model_data_dependent = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer_data_dependent = optim.SGD(model_data_dependent.parameters(), lr=0.01, momentum=0.9)
    for epsilon in epsilon_values:
        epsilon_data_dependent = adaptive_epsilon(train_loader)
        loss_data_dependent, accuracy_data_dependent = train_and_evaluate(model_data_dependent, train_loader, test_loader, optimizer_data_dependent, epsilon_data_dependent)
        accuracies_data_dependent.append(accuracy_data_dependent)
        losses_data_dependent.append(loss_data_dependent)

    # 基于模型相关性的隐私预算分配
    model_model_related = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer_model_related = optim.SGD(model_model_related.parameters(), lr=0.01, momentum=0.9)
    for epsilon in epsilon_values:
        epsilon_model_related = model_related_epsilon(model_model_related)
        loss_model_related, accuracy_model_related = train_and_evaluate(model_model_related, train_loader, test_loader, optimizer_model_related, epsilon_model_related)
        accuracies_model_related.append(accuracy_model_related)
        losses_model_related.append(loss_model_related)

        # 结合两种隐私预算分配方法
        model_combined = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer_combined = optim.SGD(model_combined.parameters(), lr=0.01, momentum=0.9)
        for epsilon in epsilon_values:
            epsilon_data_dependent = adaptive_epsilon(train_loader)
            epsilon_model_related = model_related_epsilon(model_combined)
            # 动态调整epsilon值，这里我们尝试使用两者的平均值
            epsilon_combined = (epsilon_data_dependent + epsilon_model_related) / 2
            loss_combined, accuracy_combined = train_and_evaluate(model_combined, train_loader, test_loader,
                                                                  optimizer_combined, epsilon_combined)
            accuracies_combined.append(accuracy_combined)
            losses_combined.append(loss_combined)

    # FedAvg算法
    model_fedavg = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer_fedavg = optim.SGD(model_fedavg.parameters(), lr=0.01, momentum=0.9)
    for epsilon in epsilon_values:
        loss_fedavg, accuracy_fedavg = fedavg_train(model_fedavg, train_loader, test_loader, optimizer_fedavg)
        accuracies_fedavg.append(accuracy_fedavg)
        losses_fedavg.append(loss_fedavg)

    # 绘制结果
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epsilon_values, accuracies_no_dp, marker='o', label='No DP')
    plt.plot(epsilon_values, accuracies_data_dependent, marker='o', label='Data-Dependent DP')
    plt.plot(epsilon_values, accuracies_model_related, marker='o', label='Model-Related DP')
    plt.plot(epsilon_values, accuracies_combined, marker='o', label='Combined DP')
    plt.plot(epsilon_values, accuracies_fedavg, marker='o', label='FedAvg')
    plt.xlabel('Epsilon')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs. Epsilon')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values, losses_no_dp, marker='o', label='No DP')
    plt.plot(epsilon_values, losses_data_dependent, marker='o', label='Data-Dependent DP')
    plt.plot(epsilon_values, losses_model_related, marker='o', label='Model-Related DP')
    plt.plot(epsilon_values, losses_combined, marker='o', label='Combined DP')
    plt.plot(epsilon_values, losses_fedavg, marker='o', label='FedAvg')
    plt.xlabel('Epsilon')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epsilon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('privacy_budget_comparison_cifar.png')  # 保存图片
    plt.show()
if __name__ == "__main__":
    main()

