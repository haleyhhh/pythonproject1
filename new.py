# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # 更新全连接层的输入特征数
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 更新展平的维度
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

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

# 主函数
def main():
    epochs = 10
    epsilon_values = np.linspace(0, 1, 11)
    results = {
        'No DP': {'accuracies': [], 'losses': []},
        'Data-Dependent DP': {'accuracies': [], 'losses': []},
        'Model-Related DP': {'accuracies': [], 'losses': []},
        'Combined DP': {'accuracies': [], 'losses': []},
        'FedAvg': {'accuracies': [], 'losses': []}
    }

    model = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for strategy, values in results.items():
        for epsilon in epsilon_values:
            if strategy == 'Data-Dependent DP':
                epsilon = adaptive_epsilon(train_loader)
            elif strategy == 'Model-Related DP':
                epsilon = model_related_epsilon(model)
            elif strategy == 'Combined DP':
                epsilon = (adaptive_epsilon(train_loader) + model_related_epsilon(model)) / 2
            loss, accuracy = train_and_evaluate(model, train_loader, test_loader, optimizer, epsilon)
            values['accuracies'].append(accuracy)
            values['losses'].append(loss)

    # 绘制结果
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    for strategy, values in results.items():
        plt.plot(epsilon_values, values['accuracies'], marker='o', label=strategy)
    plt.xlabel('Epsilon')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs. Epsilon')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for strategy, values in results.items():
        plt.plot(epsilon_values, values['losses'], marker='o', label=strategy)
    plt.xlabel('Epsilon')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epsilon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('privacy_budget_comparison_mnist.png')
    plt.show()

if __name__ == "__main__":
    main()