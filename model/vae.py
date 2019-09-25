# -*- coding:utf-8 -*-
# @Author   : LuoJiahuan
# @File     : vae.py 
# @Time     : 2019/9/25 20:57


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class FC_VAE(nn.Module):
    def __init__(self):
        super(FC_VAE, self).__init__()
        self.fc1 = nn.Linear(19200, 400)

        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 19200)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 19200))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CNN_VAE(nn.Module):
    def __init__(self):
        super(CNN_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.fc1 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(1024, 32)
        self.fc3 = nn.Linear(32, 1024)

    # def encode(self, x):
    #     h1 = F.relu(self.fc1(x))
    #     return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2()
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z)
        return self.decode(z), mu, logvar