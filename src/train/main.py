'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-04-24 18:46:29
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-07 23:54:27
FilePath: \Bottles-and-Cans-classifier\src\train\main.py
Description:
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import yaml
from torch.utils.data import DataLoader,random_split

TRAINING_CONFIG = 'config/training.yaml'
# load training config
CONFIG = yaml.safe_load(open(TRAINING_CONFIG, 'r'))

# load dataset by using torchvision
transform = torchvision.transforms.Compose([
    # resize to 128x128
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()
])
# target_transform -> one-hot encoding
target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
])
dataset = torchvision.datasets.ImageFolder(root=CONFIG['data_loc']['root'], transform=transform, target_transform=target_transform)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for x, y in data_loader:
    print(x.shape, y.shape)
    print(y)
    break

# define model
from torchvision.models import regnet_y_32gf
class RegNetY32GF(nn.Module):
    def __init__(self):
        super(RegNetY32GF, self).__init__()
        self.regnet = regnet_y_32gf()
        self.regnet.fc = nn.Linear(3712, 4096)
        self.last_layer = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.regnet(x)
        return self.last_layer(x)
    
# split dataset into train and test
n = len(dataset)
n_train = int(0.8*n)
n_test = n - n_train
train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# training

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# define model
model = RegNetY32GF().to(device)

# define loss function
criterion = nn.BCEWithLogitsLoss()

# define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# training recrod
train_loss = []
train_acc = []
test_loss = []
test_acc = []
confidence = {}

# training loop
n_epoch = 10
cnt = 0

for epoch in range(n_epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = torch.argmax(y_pred, 1)
        total += y.size(0)
        correct += (predicted == torch.argmax(y,1)).sum().item()
    train_loss.append(running_loss/total)
    train_acc.append(correct/total)
    print(f'epoch: {epoch+1}, train loss: {train_loss[-1]}, train acc: {train_acc[-1] * 100:.2f}%', end=', ')

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            predicted = torch.argmax(y_pred, 1)
            total += y.size(0)
            correct += (predicted == torch.argmax(y,1)).sum().item()
    test_loss.append(running_loss/total)
    test_acc.append(correct/total)
    print(f'test loss: {test_loss[-1]}, test acc: {test_acc[-1] * 100:.2f}%')

    # calculate confidence
    model.eval()
    confidence_list = []

    for input, target in test_loader:
        input, target = input.to(device), target.to(device)
        # 計算每個實例的預測概率
        with torch.no_grad():
            output = model(input)
            probs = F.softmax(output, dim=1)

        for i in range(len(probs)):
            confidence_list.append(probs[i][torch.argmax(target[i])].item())

    confidence[epoch] = confidence_list

confidence_list = [[] for _ in range(len(confidence[0]))]
for epoch in confidence:
    for i in range(len(confidence[epoch])):
        confidence_list[i].append(confidence[epoch][i])

confidence_list = np.array(confidence_list)

mean = np.mean(confidence_list, axis=1)
std = np.std(confidence_list, axis=1)
variability = std / mean