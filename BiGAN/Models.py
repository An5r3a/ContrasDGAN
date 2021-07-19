import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from functools import partial
import itertools
import tqdm
import time

class LinearGeneratorA(nn.Module):
    
    def __init__(self, input_dimA, output_dim,dim):
        super(LinearGeneratorA,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimA,dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, output_dim))

    def forward(self, x):
        return self.layers(x)

class LinearGeneratorB(nn.Module):

    def __init__(self, input_dimB, output_dim, dim):
        super(LinearGeneratorB,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimB,dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, output_dim))

    def forward(self, x):
        return self.layers(x)
    

class BiGANDiscriminatorA(nn.Module):
    def __init__(self,latent_dim,dim):
        super(BiGANDiscriminatorA,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim*2, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid())

    def forward(self, x,z):
        xz = torch.cat((x, z), dim=1)
        return self.layers(xz)
    
class BiGANDiscriminatorB(nn.Module):
    def __init__(self, latent_dim, dim):
        super(BiGANDiscriminatorB,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim*2, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid())

    def forward(self, x,z):
        xz = torch.cat((x, z), dim=1)
        return self.layers(xz)
    
class Classifier(nn.Module):

    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 19)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.relu(self.fc1(input))
        logits = self.fc2(x)

        return logits
