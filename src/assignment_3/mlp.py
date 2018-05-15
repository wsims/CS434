"""MLP Module

This module includes the subroutines to create multilayer perceptron networks
for all parts of assignment 3.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import cPickle
import os
import os.path
from PIL import Image

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

class CIFAR10v2(torch.utils.data.Dataset):
    """Based off of the default CIFAR10 dataset class: 

    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py

    Inputs:
        root (string): Root directory where 'cifar-10-batches-py' folder is stored.
        train (bool): If true, train dataset is created from data batch files 1-4.  If false,
            creates a test dataset from data batch 5.
        transform (callable): function that takes in a PIL image and returns a transformed version.
            For our cases, we typically use this to return a normalized tensor.
        target_transform (callable): function that takes in the target and transforms it.

    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4'
    ]

    test_list = [
        'data_batch_5'
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                file = os.path.join(self.root, self.base_folder, fentry)
                fo = open(file, 'rb')
                entry = cPickle.load(fo)
                self.train_data.append(entry['data'])
                self.train_labels += entry['labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((40000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            f = self.train_list[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            entry = cPickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = entry['labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        """
        Inputs:
            index (int): Index value

        Outputs:
            img (tensor): transformed image
            target (int): number from 0-9 corresponding to actual classification
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Returns the number of observations in the dataset"""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class SigmoidNet(nn.Module):
    """A three layer neural net that uses sigmoid as the activation function"""
    def __init__(self, dropout=0.2):
        super(SigmoidNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), 0)

class ReluNet(nn.Module):
    """A three layer neural net that uses relu as the activation function"""
    def __init__(self):
        super(ReluNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), 0)

class FourLayerNet(nn.Module):
    """A four layer neural net that uses sigmoid as the activation function"""
    def __init__(self):
        super(FourLayerNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_drop = nn.Dropout(0.2)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.sigmoid(self.fc3(x))
        x = self.fc3_drop(x)
        return F.log_softmax(self.fc4(x), 0)

def train(epoch, model, optimizer, train_loader, log_interval=100):
    """Trains the model through an entire epoch.

    Inputs:
        epoch (int): Index corresponding to the epoch number.
        model (nn.Module subclass): Neural network model object to train
        optimizer (torch.optim): PyTorch optimizer object
        train_loader (PyTorch Dataloader): loaded training data set for CIFAR10
        log_interval (int): Controls how many batches are run in between logging

    Outputs:
        avg_loss (float): Average negative log loss over the course of this epoch.
    """
    model.train()
    avg_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_loss /= len(train_loader)
    return avg_loss

def validate(loss_vector, accuracy_vector, model, validation_loader):
    """Tests the classifier against the validation set.

    Inputs:
        loss_vector (list): list of negative log losses for each validation.
            This function appends a new loss to the end of this list after running.
        accuracy_vector (list): list of classifier accuracies for each validation.
            This function appends a new accuracy to the end of this list upon completion.
        model (nn.Module subclass): Neural network model object to validate
        validation_loader (PyTorch Dataloader): loaded validation set for CIFAR10.
            Can be either the validation set or the testing set.
    """
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

def get_cifar10_data(path='.', train=True, batch_size=32):
    """Generates a dataloader object for CIFAR10 data.

    This function can generate the training set (data batches 1-4) and the
    validation set (data batch 5).

    Inputs:
        path (string): Root of directory where 'cifar-10-batches-py' directory can be found
        train (bool): If true, the returned dataloader contains the training data.  If
            false, dataloader contains the validation data.
        batch_size (int): Number of observations per batch.

    Outputs:
        loader (PyTorch Dataloader): Loader containing desired data set.
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    loader = torch.utils.data.DataLoader(
        CIFAR10v2(path, train=train,
                  transform=transforms.Compose([
                      transforms.ToTensor()
                  ])),
        batch_size=batch_size, shuffle=train, **kwargs)

    return loader

def get_cifar10_test_data(path='.', batch_size=32):
    """Generates a dataloader object for CIFAR10 data.

    This function can generate the testing set (test batch)

    Inputs:
        path (string): Root of directory where 'cifar-10-batches-py' directory can be found
        batch_size (int): Number of observations per batch.

    Outputs:
        loader (PyTorch Dataloader): Loader containing desired data set.
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(path, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    return loader
    
def sigmoid_NN_train_and_val(train_loader, validation_loader, 
                     lr=0.01, momentum=0.5, epochs=10, dropout=0.2,
                     weight_decay=0):
    """Creates and trains a three layer neural net using the sigmoid function
    as the activation function.

    Inputs:
        train_loader (PyTorch Dataloader): Loader containing the training data
        validation_loader (PyTorch Dataloader): Loader containing the validation data
        lr (float): Learning rate
        momentum (float): Momentum
        epochs (int): Number of epochs to train classifier on.
        dropout (float): Dropout rate for neural network.
        weight_decay (float): Weight decay for neural network.

    Outputs:
        train_loss (list): list of average negative log loss on the training data for each epoch
        accv (list): List of validation accuracy on the validation data for each epoch.
        model (Neural Network object): Final, trained neural network object
    """
    model = SigmoidNet(dropout)
    if cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)

    print(model)

    lossv, accv, train_loss = [], [], []
    for epoch in range(1, epochs + 1):
        train_loss.append(train(epoch, model, optimizer, train_loader))
        validate(lossv, accv, model, validation_loader)

    return train_loss, accv, model

def relu_NN_train_and_val(train_loader, validation_loader, 
                     lr=0.01, momentum=0.5, epochs=10):
    """Creates and trains a three layer neural net using the relu function
    as the activation function.

    Inputs:
        train_loader (PyTorch Dataloader): Loader containing the training data
        validation_loader (PyTorch Dataloader): Loader containing the validation data
        lr (float): Learning rate
        momentum (float): Momentum
        epochs (int): Number of epochs to train classifier on.

    Outputs:
        train_loss (list): list of average negative log loss on the training data for each epoch
        accv (list): List of validation accuracy on the validation data for each epoch.
        model (Neural Network object): Final, trained neural network object
    """
    model = ReluNet()
    if cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(model)

    lossv, accv, train_loss = [], [], []
    for epoch in range(1, epochs + 1):
        train_loss.append(train(epoch, model, optimizer, train_loader))
        validate(lossv, accv, model, validation_loader)

    return train_loss, accv, model

def four_NN_train_and_val(train_loader, validation_loader, 
                     lr=0.01, momentum=0.5, epochs=10):
    """Creates and trains a four layer neural net using the sigmoid function
    as the activation function.

    Inputs:
        train_loader (PyTorch Dataloader): Loader containing the training data
        validation_loader (PyTorch Dataloader): Loader containing the validation data
        lr (float): Learning rate
        momentum (float): Momentum
        epochs (int): Number of epochs to train classifier on.

    Outputs:
        train_loss (list): list of average negative log loss on the training data for each epoch
        accv (list): List of validation accuracy on the validation data for each epoch.
        model (Neural Network object): Final, trained neural network object
    """
    model = FourLayerNet()
    if cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(model)

    lossv, accv, train_loss = [], [], []
    for epoch in range(1, epochs + 1):
        train_loss.append(train(epoch, model, optimizer, train_loader))
        validate(lossv, accv, model, validation_loader)

    return train_loss, accv, model
