import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt

from results.MNIST.pr import federated_accuracies


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
server_model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(server_model.parameters(), lr=0.01, momentum=0.9)

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  # 修改batch_size

# 训练模型的函数
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader) / epochs

# 评估模型的函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# 训练和评估服务器模型
initial_train_loss = train_model(server_model, train_loader, criterion, optimizer, epochs=5)
initial_accuracy = evaluate_model(server_model, test_loader)
print(f"Initial Training Loss: {initial_train_loss}")
print(f"Initial Test Accuracy: {initial_accuracy}")

# 定义剪枝函数
def prune_model(model, prune_amount):
    for name, param in model.named_parameters():
        if 'conv' in name:
            threshold = torch.quantile(torch.abs(param), prune_amount)
            mask = torch.abs(param) > threshold
            param.data.mul_(mask)

# 普通剪枝
prune_model(server_model, prune_amount=0.9)  # 使用较大的剪枝比例
pruned_accuracy = evaluate_model(server_model, test_loader)
print(f"Test Accuracy after pruning: {pruned_accuracy}")

# 微调服务器模型
optimizer = optim.SGD(server_model.parameters(), lr=0.0001, momentum=0.9)  # 减小学习率
normal_accuracies = [pruned_accuracy]
for epoch in range(5):  # 减少微调轮数以突出联邦剪枝的优势
    train_model(server_model, train_loader, criterion, optimizer, epochs=1)
    accuracy = evaluate_model(server_model, test_loader)
    normal_accuracies.append(accuracy)
    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

# 创建多个客户端模型并训练
client_models = [copy.deepcopy(server_model) for _ in range(10)]
for model in client_models:
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_model(model, train_loader, criterion, optimizer, epochs=5)

# 联邦剪枝
def federated_pruning(models, prune_amount=0.2):  # 初始使用较小的剪枝比例
    for model in models:
        for name, param in model.named_parameters():
            if 'conv' in name:
                threshold = torch.quantile(torch.abs(param), prune_amount)
                mask = torch.abs(param) > threshold
                param.data.mul_(mask)

# 微调联邦剪枝模型
def fine_tune_federated(models, train_loader, criterion, initial_lr=0.01, epochs=20):
    for model in models:
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
        # 学习率预热
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)
        for epoch in range(epochs):
            train_model(model, train_loader, criterion, optimizer, epochs=1)
            scheduler.step()  # 更新学习率
            accuracy = evaluate_model(model, test_loader)
            print(f'Epoch {epoch+1}, Federated Average Accuracy: {accuracy:.4f}')

# 调用联邦剪枝和微调函数
federated_pruning(client_models, prune_amount=0.2)  # 使用较小的剪枝比例
fine_tune_federated(client_models, train_loader, criterion, initial_lr=0.01, epochs=20)

# 聚合客户端模型
aggregated_model = copy.deepcopy(client_models[0])
for model in client_models[1:]:
    for name, param in model.named_parameters():
        aggregated_model.state_dict()[name] += param.data
for name, param in aggregated_model.named_parameters():
    param.data.div_(len(client_models))

# 评估聚合后的模型
aggregated_accuracy = evaluate_model(aggregated_model, test_loader)
print(f"Aggregated test accuracy: {aggregated_accuracy}")

# 绘制剪枝前后的准确率对比
plt.figure(figsize=(10, 6))
epochs = range(1, len(normal_accuracies) + 1)
plt.plot(epochs, normal_accuracies, marker='o', color='red', label='Normal Pruning')
federated_epochs = range(1, len(federated_accuracies) + 1)
plt.plot(federated_epochs, federated_accuracies, marker='s', color='blue', label='Federated Pruning', linestyle='--')
plt.plot([len(federated_epochs)], [aggregated_accuracy], color='green', marker='x', label='Aggregated Federated Pruning')

plt.legend()
plt.ylabel('Test Accuracy')
plt.xlabel('Epochs')
plt.title('Comparison of Normal and Federated Pruning')
plt.grid(True)
plt.ylim(0.7, 1.0)
plt.xlim(0, len(federated_epochs) + 1)

save_path = 'pruning_comparison_cifar10.png'
plt.savefig(save_path)
print(f'Image saved to {save_path}')

try:
    plt.show()
except Exception as e:
    print(f"Failed to display image: {e}")
    print("You can check the saved image in the current directory.")