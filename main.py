import torch
import torch.nn as nn
from model import *
from loader import CustomLoader as cl
from torch.utils.data import DataLoader
from show import show
import random
torch.manual_seed(1234)

BATCH = 4
EPOCHS = 2450
LR = 0.05

trainset = cl('./data/heart_processed-train.csv')
trainloader = DataLoader(dataset=trainset, batch_size=BATCH)
testset = cl('./data/heart_processed-test.csv')
testloader = DataLoader(dataset=testset, batch_size=BATCH)

network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=LR)


train(network,optimizer, criterion, trainloader, testloader, EPOCHS)
