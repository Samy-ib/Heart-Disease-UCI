import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13,70)
        self.fc2 = nn.Linear(70,2)
        # self.fc3 = nn.Linear(35, 2)
        # self.fc4 = nn.Linear(16, 2)

        self.dropout = nn.Dropout(p=0.0)
    
    def forward(self, X):
        X = self.dropout(F.relu(self.fc1(X)))
        # X = self.dropout(F.relu(self.fc2(X)))
        # X = self.dropout(F.relu(self.fc3(X)))
        X = self.fc2(X)

        return F.softmax(X, dim=0)

def train(network, optimizer, criterion, trainloader, testloader, EPOCHS):
    train_log=[]
    test_log=[]
    last_loss = 10
    for epoch in range(EPOCHS):
        training_loss = 0
        network.train() #Set the network to training mode
        for X, Y in trainloader:
            optimizer.zero_grad()
            out = network(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(trainloader) 
        testing_loss, accuracy = test(network, criterion, testloader)

        log(EPOCHS, epoch, training_loss, testing_loss, accuracy)
        last_loss = checkpoint(network, last_loss, testing_loss)

        train_log.append(training_loss)
        test_log.append(testing_loss)
    show(train_log, test_log)

def test(network, criterion, testloader):
    testing_loss = 0
    accuracy = 0
    with torch.no_grad(): #Desactivate autograd engine (reduce memory usage and speed up computations)
        network.eval() #set the layers to evaluation mode(batchnorm and dropout)
        for X, Y in testloader:
            out = network(X)
            loss = criterion(out, Y)
            testing_loss += loss.item()


            predict = network(X)
            _, predict_y = torch.max(predict, 1)

            accuracy = accuracy + (torch.sum(Y==predict_y).float())
        
        return testing_loss/len(testloader), 100*accuracy/(len(testloader)*len(Y))

def log(epochs, epoch, trainL, testL, acc):
    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
        "Training Loss: {:.3f}.. ".format(trainL),
        "Testing Loss: {:.3f}.. ".format(testL),
        "Test Accuracy: {:.3f}".format(acc))

def checkpoint(network, last, actual):
    '''
        Saves the model if the loss decrease
    '''
    if actual < last :
        print("Loss decreased, saving the model..")
        torch.save(network, 'model.pt')
        return actual
    else:
        return last

def show(trainL, testL):
    plt.plot(trainL, label='Training loss')
    plt.plot(testL, label='Testing Loss')
    plt.legend(frameon=False)
    plt.show()