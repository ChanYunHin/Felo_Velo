import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import os

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# train_data = CIFAR10(
#     root = "data",
#     train = True,
#     download = True,
#     transform = img_transform
# )
# test_data = CIFAR10(
#     root = "data",
#     train = False,
#     download = True,
#     transform = img_transform
# )

# dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.n_classes = 10


#         self.label_embedding = nn.Embedding(self.n_classes, 256)
        

#         # Encoder
#         self.img_encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#         )
        
#         self.img_encoder_1 = nn.Sequential(
#             nn.Linear(8 * 8 * 16, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#         )
        

#         self.fc_label_img = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             # nn.Linear(512, 512),
#             # nn.BatchNorm1d(512),
#             # nn.ReLU()
#         )

#         self.fc21 = nn.Linear(512, 512)
#         self.fc22 = nn.Linear(512, 512)

#         # Decoder
#         self.fc3 = nn.Linear(768, 512)
#         self.fc_bn3 = nn.BatchNorm1d(512)
#         self.fc4 = nn.Linear(512, 8 * 8 * 16)
#         self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

#         self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.bn5 = nn.BatchNorm2d(32)
#         self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn6 = nn.BatchNorm2d(32)
#         self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.bn7 = nn.BatchNorm2d(16)
#         self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

#         self.relu = nn.ReLU()

#     def encode(self, x, labels):
#         labels = self.label_embedding(labels)
#         x = self.img_encoder(x).view(-1, 8 * 8 * 16)
#         x = self.img_encoder_1(x)
#         concat_vec = torch.cat([x, labels], dim=1)
#         concat_vec = self.fc_label_img(concat_vec)

#         return self.fc21(concat_vec), self.fc22(concat_vec), labels

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         if torch.cuda.is_available():
#             eps = torch.cuda.FloatTensor(std.size()).normal_()
#         else:
#             eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)

#     def decode(self, z, labels):
#         z = torch.cat([z, labels], dim=1)
#         fc3 = self.relu(self.fc_bn3(self.fc3(z)))
#         fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

#         conv5 = self.relu(self.bn5(self.conv5(fc4)))
#         conv6 = self.relu(self.bn6(self.conv6(conv5)))
#         conv7 = self.relu(self.bn7(self.conv7(conv6)))
#         return self.conv8(conv7).view(-1, 3, 32, 32)

#     def forward(self, x, labels):
#         mu, logvar, labels = self.encode(x, labels)
#         z = self.reparametrize(mu, logvar)
#         return self.decode(z, labels), mu, logvar

latent_size = 32
label_embedding_size = 512
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# In this experiment, smaller loss doesn't mean better generative capacity.
# In fact, no convergence model has a better generative ability. (see the generated image)
# So We could train the VAE just in a few epoches?
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.n_classes = 10


        self.label_embedding = nn.Embedding(self.n_classes, label_embedding_size)
        

        self.fc_label_img = nn.Sequential(
            nn.Linear(784 + label_embedding_size, latent_size),
            # nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU()
        )

        self.fc21 = nn.Linear(latent_size, latent_size)
        self.fc22 = nn.Linear(latent_size, latent_size)

        # Decoder
        # input dimension = latent dimension + label dimension
        self.decoder = nn.Sequential(
            nn.Linear(label_embedding_size + latent_size, latent_size),
            # nn.BatchNorm1d(latent_size),
            nn.Linear(latent_size, 784),
            nn.Sigmoid()
        )
        



        # self.relu = nn.ReLU()

    def encode(self, x, labels):
        labels = self.label_embedding(labels)
        concat_vec = torch.cat([x, labels], dim=1)
        concat_vec = self.fc_label_img(concat_vec)
        return self.fc21(concat_vec), self.fc22(concat_vec), labels

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def decode(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        return self.decoder(z)

    def forward(self, x, labels):
        mu, logvar, labels = self.encode(x, labels)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def generate_img(self, z, labels):
        labels = self.label_embedding(labels)
        return self.decode(z, labels)




model = VAE()
if torch.cuda.is_available():
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        img, y_label = data
        img = img.view(img.size(0), -1)
        # img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            y_label = y_label.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img, y_label)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.item() / len(img)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    if epoch % 10 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))


    if epoch <= 20 or epoch % 10 == 0:
        with torch.no_grad():
            for i in range(10):
                eps = torch.randn((batch_size, latent_size)).cuda()
                fake_label = torch.full((batch_size, ), fill_value=i, dtype=torch.int32).cuda()
                generate_img = model.generate_img(eps, fake_label)
                save = to_img(generate_img.cpu().data)
                if not os.path.exists('./vae_img/{}/'.format(i)):
                    os.mkdir('./vae_img/{}/'.format(i))
                save_image(save, './vae_img/{}/generated_image_{}.png'.format(i, epoch))

# torch.save(model.state_dict(), './vae.pth')



# eps = torch.randn((batch_size, latent_size)).cuda()
# fake_label = torch.ones((batch_size), dtype=torch.int32).cuda()
# generate_img = model.generate_img(eps, fake_label)
# save = to_img(generate_img.cpu().data)
# save_image(save, './vae_img/generated_image_{}.png'.format(1))
