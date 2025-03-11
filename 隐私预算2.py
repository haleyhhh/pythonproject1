import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # 输入特征图的大小为 14x14
        self.fc2 = nn.Linear(128, 10)  # MNIST 有 10 个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 输出特征图大小为 14x14
        x = x.view(-1, 32 * 14 * 14)  # 将特征图展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 初始化模型和优化器
def init_model_optimizer():
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, optimizer


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


# 训练和评估模型的函数
def train_and_evaluate_model(model, optimizer, criterion, train_loader, test_loader, epochs, noise_multiplier):
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            noise = torch.randn_like(output) * noise_multiplier
            loss = criterion(output + noise, target)
            loss.backward()
            optimizer.step()

    model.eval()
    accuracy = evaluate_model(model, test_loader)
    return accuracy


# 逐步减少噪声的训练和评估模型的函数
def train_and_evaluate_model_noafl(model_noafl, optimizer_noafl, criterion, train_loader, test_loader, epochs,
                                   noise_scale):
    for epoch in range(epochs):
        model_noafl.train()
        for data, target in train_loader:
            optimizer_noafl.zero_grad()
            output = model_noafl(data)
            # 逐步减少噪声
            noise_multiplier = noise_scale / (1 + epoch * 0.1)
            noise = torch.randn_like(output) * noise_multiplier
            loss = criterion(output + noise, target)
            loss.backward()
            optimizer_noafl.step()

        # 学习率衰减
        for param_group in optimizer_noafl.param_groups:
            param_group['lr'] *= 0.95

    model_noafl.eval()
    accuracy = evaluate_model(model_noafl, test_loader)
    return accuracy


# 主函数
def main():
    global accuracies_dp_fl, accuracies_ldp_fl, accuracies_noafl

    epsilons = np.linspace(0.1, 8, 20)
    accuracies_dp_fl = np.zeros(len(epsilons))
    accuracies_ldp_fl = np.zeros(len(epsilons))
    accuracies_noafl = np.zeros(len(epsilons))

    for i, eps in enumerate(epsilons):
        model_dp, optimizer_dp = init_model_optimizer()
        model_ldp, optimizer_ldp = init_model_optimizer()
        model_noafl, optimizer_noafl = init_model_optimizer()

        criterion = nn.CrossEntropyLoss()

        # 训练并评估 DP-FL 模型
        accuracies_dp_fl[i] = train_and_evaluate_model(model_dp, optimizer_dp, criterion, train_loader, test_loader, 10,
                                                       1 / eps)

        # 训练并评估 LDP-FL 模型
        accuracies_ldp_fl[i] = train_and_evaluate_model(model_ldp, optimizer_ldp, criterion, train_loader, test_loader,
                                                        10, 1 / (eps * 0.5))

        # 训练并评估 NoAFL/DPBA 模型，逐步减少噪声
        accuracies_noafl[i] = train_and_evaluate_model_noafl(model_noafl, optimizer_noafl, criterion, train_loader,
                                                             test_loader, 10, 1 / eps)

    # 绘制结果
    plt.plot(epsilons, accuracies_dp_fl, 'r-^', label='DP-FL')
    plt.plot(epsilons, accuracies_ldp_fl, 'b-o', label='LDP-FL')
    plt.plot(epsilons, accuracies_noafl, 'g-s', label='NoAFL/DPBA')
    plt.xlabel('Privacy Budget (epsilon)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Privacy Budget on MNIST')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()