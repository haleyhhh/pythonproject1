import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 62)  # 更新为62个输出

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
server_model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(server_model.parameters(), lr=0.01, momentum=0.9)

# 加载FEMNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 训练模型的函数
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

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
train_loss = train_model(server_model, train_loader, criterion, optimizer)
test_accuracy_before = evaluate_model(server_model, test_loader)
print(f'Training loss: {train_loss}')
print(f'Test accuracy before pruning: {test_accuracy_before}')

# 定义剪枝函数
def prune_model(model, prune_amount):
    for name, param in model.named_parameters():
        if 'conv' in name:
            threshold = torch.quantile(torch.abs(param), 1 - prune_amount)
            mask = torch.abs(param) > threshold
            param.data.mul_(mask)

# 普通剪枝
prune_model(server_model, prune_amount=0.7)  # 使用较大的剪枝比例
test_accuracy_after_pruning = evaluate_model(server_model, test_loader)
print(f'Test accuracy after pruning: {test_accuracy_after_pruning}')

# 微调服务器模型
optimizer = optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
normal_accuracies = [test_accuracy_after_pruning]
for epoch in range(3):  # 微调轮数
    train_model(server_model, train_loader, criterion, optimizer)
    accuracy = evaluate_model(server_model, test_loader)
    normal_accuracies.append(accuracy)
    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

test_accuracy_after_finetuning = normal_accuracies[-1]
print(f'Test accuracy after finetuning: {test_accuracy_after_finetuning}')

# 创建多个客户端模型并训练
client_models = [copy.deepcopy(server_model) for _ in range(10)]
for model in client_models:
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_model(model, train_loader, criterion, optimizer)

# 联邦剪枝
def federated_pruning(models, prune_amount=0.2):  # 使用较小的剪枝比例
    global_thresholds = {}
    for model in models:
        for name, param in model.named_parameters():
            if 'conv' in name:
                global_thresholds[name] = torch.quantile(torch.abs(param), prune_amount)
    for model in models:
        for name, param in model.named_parameters():
            if 'conv' in name:
                threshold = global_thresholds[name]
                mask = torch.abs(param) > threshold
                param.data.mul_(mask)

federated_pruning(client_models, prune_amount=0.2)  # 使用较小的剪枝比例

# 微调联邦剪枝模型
federated_accuracies = []
for epoch in range(10):  # 增加微调轮数
    fed_accuracies = []
    for model in client_models:
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        train_model(model, train_loader, criterion, optimizer)
        accuracy = evaluate_model(model, test_loader)
        fed_accuracies.append(accuracy)
    average_accuracy = sum(fed_accuracies) / len(fed_accuracies)
    federated_accuracies.append(average_accuracy)
    print(f'Epoch {epoch+1}, Federated Average Accuracy: {average_accuracy:.4f}')

# 聚合客户端模型
aggregated_model = copy.deepcopy(client_models[0])
for model in client_models[1:]:
    for name, param in model.named_parameters():
        aggregated_model.state_dict()[name] += param.data
for name, param in aggregated_model.named_parameters():
    param.data.div_(len(client_models))

# 评估聚合后的模型
aggregated_accuracy = evaluate_model(aggregated_model, test_loader)
print(f'Aggregated test accuracy: {aggregated_accuracy}')

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
plt.ylim(0.8, 1.0)
plt.xlim(0, len(federated_epochs) + 1)

# 定义保存图片的路径
save_path = 'pruning_comparison_feminist.png'
print(f'Attempting to save image to: {save_path}')

# 尝试保存图片
try:
    plt.savefig(save_path)
    print(f'Image successfully saved to: {save_path}')
except Exception as e:
    print(f"Failed to save image: {e}")
    print("Check if the directory is writable or try a different directory.")

try:
    plt.show()
except Exception as e:
    print(f"Failed to display image: {e}")
    print("You can check the saved image in the current directory.")