import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# class Discriminator(nn.Module):
#     leak = 0.1

#     class Block(nn.Module):
#         def __init__(self, inch, outch):
#             super(Discriminator.Block, self).__init__()
#             self.layer1 = nn.Sequential(
#                 nn.utils.spectral_norm(nn.Conv2d(inch, outch, 3, stride=1, padding=1)),
#                 nn.LeakReLU(Discriminator.leak))
#             self.layer2 = nn.Sequential(
#                 nn.utils.spectral_norm(nn.Conv2d(outch, outch, 3, stride=1, padding=1)),
#                 nn.LeakyReLU(Discriminator.leak))
        
#         def forward(self, x):
#             x = self.layer1(x)
#             x = self.layer2(x)
#             return x

#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.block1 = self.Block(1, 32) # grey
#         self.pool1 = nn.MaxPool2d(2)
#         self.block2 = self.Block(32, 64)
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.utils.spectral_norm(nn.Linear(7 * 7 * 64, 256))
#         self.lrelu = nn.LeakyReLU(Discriminator.leak)
#         self.fc2 = nn.utils.spectral_norm(nn.Linear(256, 1)) # real or fake

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.pool1(x)
#         x = self.block2(x)
#         x = self.pool2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.lrelu(x)
#         x = self.fc2(x)
#         return x
class Discriminator(nn.Module):
    leak = 0.1
    l_dim = 10

    class Embed(nn.Module):
        def __init__(self):
            super(Discriminator.Embed, self).__init__()
            self.layer = nn.utils.spectral_norm(
                nn.Linear(Discriminator.l_dim, 128 * 7 * 7))
        
        def forward(self, l):
            return self.layer(l)

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, 3, stride=1, padding=1))
        self.layer2 = nn.utils.spectral_norm(nn.Conv2d(32, 32, 4, stride=2, padding=1))
        self.layer3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.layer4 = nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, stride=2, padding=1))
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        self.fc = nn.utils.spectral_norm(nn.Linear(128 * 7 * 7, 1))
        self.embed = self.Embed()

    def forward(self, x, l):
        m = x
        m = nn.LeakyReLU(Discriminator.leak)(self.layer1(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer2(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer3(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer4(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer5(m))
        m = m.view(-1, 128 * 7 * 7)
        e = self.embed(l)
        return self.fc(m) + torch.sum(m * e)

# class Generator(nn.Module):
#     leak = 0.1
#     z_dim = 128

#     class Block(nn.Module):
#         def __init__(self, inch, outch):
#             super(Generator.Block, self).__init__()
#             self.layer1 = nn.Sequential(
#                 nn.Conv2d(inch, outch, 3, stride=1, paddiing=1),
#                 nn.BatchNorm2d(outch),
#                 nn.LeakyReLU(Generator.leak))
#             self.layer2 = nn.Sequential(
#                 nn.Conv2d(outch, outch, 3, stride=1, paddiing=1),
#                 nn.BatchNorm2d(outch),
#                 nn.LeakyReLU(Generator.leak))

#         def forward(self, x):
#             x = self.layer1(x)
#             x = self.layer2(x)
#             return x

#         def __init__(self):
#             super(Generator, self).__init__()
#             self.fc = nn.Linear(Geneartor.z_dim, 64 * 7 * 7)
#             self.block1 = self.Block(64, 32)
#             self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
#             self.block2 = self.Block(32, 32)
#             self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
#             self.conv = nn.Conv2d(32, 1, 3, stride=1, padding=1)
#             self.tanh = nn.Tanh()

#         def forward(self, z):
#             x = self.fc(z)
#             x = x.view(-1, 64, 7, 7)
#             x = self.block1(x)
#             x = self.upsample1(x)
#             x = self.block2(x)
#             x = self.upsample2(x)
#             x = self.conv(x)
#             x = self.tanh(x)
#             return x
class Generator(nn.Module):
    z_dim = 128
    l_dim = 10
    leak = 0.1

    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(Generator.z_dim + Generator.l_dim, 128 * 7 * 7)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(Generator.leak),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(Generator.leak),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, l):
        m = torch.cat([z, l], axis=1)
        m = self.fc(m)
        m = m.view(-1, 128, 7, 7)
        m = self.model(m)
        return m

def save_images(data, rows, cols, fpath):
    images = data.numpy().reshape(-1, 28, 28)

    fig, axes = plt.subplots(rows.cols, figsize=(5, 5))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margin(0, 0)
    for ax, i in zip(axes.flat, range(rows * cols)):
        ax.axis('off')
        ax.imshow(images[i], aspect='auto')
    #plt.show()
    plt.savefig(fpath)
    plt.close()

def test(device, gen, batch_size):
    gen.to(device)

    Tensor = torch.cuda.FloatTensor if device is 'cuda' else torch.FloatTensor
    target = Tensor(np.arange(batch_size))
    l = torch.zeros(target.shape[0], 10)

def train(device, dis, gen, data_loader, batch_size, epochs):
    d_lr = 4e-4
    g_lr = 1e-4

    dis.to(device)
    gen.to(device)

    d_opt = torch.optim.Adam(dis.parameters(), lr=d_lr)
    g_opt = torch.optim.Adam(gen.parameters(), lr=g_lr)
    d_lrs = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=0.99)
    g_lrs = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=0.99)

    Tensor = torch.cuda.FloatTensor if device is 'cuda' else torch.FloatTensor
    for epoch in range(1, epochs + 1):
        batches_done = 0
        for i, (images, target) in enumerate(data_loader):
            real = images.type(Tensor)

            # make one-hot tensor
            l = torch.zeros(target.shape[0], 10).type(Tensor)
            l[range(l.shape[0]), target] = 1

            # train D
            z = Tensor(np.random.normal(0, 1, (images.shape[0], Generator.z_dim)))
            fake = gen(z, l)
            d_opt.zero_grad() # torch accumulates grad so clear it
            d_loss = torch.mean(F.relu(1.0 - dis(real, l))) \
                + torch.mean(F.relu(1.0 + dis(fake.detach(), l))) # WGAN Hinge loss
            d_loss.backward()
            d_opt.step()

            # train G
            g_opt.zero_grad()
            g_loss = -1.0 * torch.mean(dis(fake, l))
            g_loss.backward()
            g_opt.step()

            if batches_done % 300 == 0:
                print(f'Epoch {epoch}/{epochs} Batch {batches_done}/{len(data_loader)} \
                    d_loss {d_loss.item():.4f} g_loss {g_loss.item():.4f}')
            batches_done += 1
        d_lrs.step()
        g_lrs.step()