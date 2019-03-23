import torch
import torch.nn as nn
from model import *
from loader import CustomLoader as cl
from torch.utils.data import DataLoader
from show import show
import random
torch.manual_seed(1234)

BATCH = 8
EPOCHS = 1000
LR = 0.0005

trainset = cl('./data/heart_processed-train.csv')
trainloader = DataLoader(dataset=trainset, batch_size=BATCH)
testset = cl('./data/heart_processed-test.csv')
testloader = DataLoader(dataset=testset, batch_size=BATCH)

network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=LR)


train(network,optimizer, criterion, trainloader, testloader, EPOCHS)




# for epoch in range(EPOCHS):
#     training_loss = 0
#     testing_loss = 0
#     accuracy = 0
#     network.train()
#     for X, Y in trainloader:
#         optimizer.zero_grad()
#         out = network(X)
#         loss = criterion(out, Y)
#         loss.backward()
#         optimizer.step()

#         training_loss += loss.item()

    
#     # print('L:', len(trainloader))
#     print('Y:',Y.data)

#     # print("epoch : ", epoch+1,"loss : ", training_loss/len(trainloader))
#     predict = network(X)
#     _, predict_y = torch.max(predict, 1)
#     print('P:',predict_y)

#     print('S', 100*torch.sum(Y==predict_y)/3)