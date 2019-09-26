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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=17920, out_channels=128, kernel_size=(15, 20), stride=2),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=(1, 1),
                               output_padding=(1, 1)),
            nn.ReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # self.deconv1 = nn.Sequential(
        #     nn.ConvTranspose2d()
        # )
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        # )

        # self.decode = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=12288, out_channels=256, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=[3,4], stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=[3,4], stride=2),
        #     nn.Sigmoid(),
        # )

        self.fc1 = nn.Linear(17920, 32)
        self.fc2 = nn.Linear(17920, 32)
        self.fc3 = nn.Linear(32, 17920)

    def encoder(self, x):
        h1 = self.conv1(x)  # (1, 32, 60, 80)
        h2 = self.conv2(h1)  # (1, 64, 30, 40)
        h3 = self.conv3(h2)  # (1, 128, 15, 20)
        h4 = self.conv4(h3)  # (1, 256, 7, 10)
        h = h4.view(h4.size(0), -1)
        return h

    def decoder(self, x):
        z1 = self.deconv1(x)
        z2 = self.deconv2(z1)
        z3 = self.deconv3(z2)
        z4 = self.deconv4(z3)
        return z4

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def decode(self, z):
    #    h3 = F.relu(self.fc3(z))
    #    return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z)
        z = z.view(z.size(0), 17920, 1, 1)
        print(z.shape)
        return self.decoder(z), mu, logvar
