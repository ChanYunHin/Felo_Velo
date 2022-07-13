import torch
from torch.nn.modules.activation import Sigmoid
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
import torchvision.models as models
import os

from parameters import args 

# if not os.path.exists('./vae_img'):
#     os.mkdir('./vae_img')

# #用上采样加卷积代替了反卷积
# class ResizeConv2d(nn.Module): 

#     def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.mode = mode
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
#         x = self.conv(x)
#         return x


# class ResNet18Enc(nn.Module):
#     def __init__(self, z_dim=32):
#         super(ResNet18Enc, self).__init__()
#         self.z_dim = z_dim
#         self.ResNet18 = models.resnet18(pretrained=True)
#         self.num_feature = self.ResNet18.fc.in_features
#         self.ResNet18.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

#     def forward(self, x):
#         x = self.ResNet18(x)
#         mu = x[:, :self.z_dim]
#         logvar = x[:, self.z_dim:]
#         return mu, logvar


# class BasicBlockDec(nn.Module):

#     def __init__(self, in_planes, stride=1):
#         super().__init__()

#         planes = int(in_planes / stride)

#         self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_planes)

#         if stride == 1:
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential()
#         else:
#             self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential(
#                 ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = torch.relu(self.bn2(self.conv2(x)))
#         out = self.bn1(self.conv1(out))
#         out += self.shortcut(x)
#         out = torch.relu(out)
#         return out


# class ResNet18Dec(nn.Module):

#     def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3):
#         super().__init__()
#         self.in_planes = 512

#         self.linear = nn.Linear(z_dim, 512)

#         self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
#         self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
#         self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
#         self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
#         self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

#     def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
#         strides = [stride] + [1] * (num_Blocks - 1)
#         layers = []
#         for stride in reversed(strides):
#             layers += [BasicBlockDec(self.in_planes, stride)]
#         self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, z):
#         x = self.linear(z)
#         x = x.view(z.size(0), 512, 1, 1)
#         x = F.interpolate(x, scale_factor=7)
#         x = self.layer4(x)
#         x = self.layer3(x)
#         x = self.layer2(x)
#         x = self.layer1(x)
#         x = F.interpolate(x, size=(112, 112), mode='bilinear')
#         x = torch.sigmoid(self.conv1(x))
#         x = x.view(x.size(0), 3, 224, 224)
#         return x


# class ServerModel(nn.Module):
#     def __init__(self, z_dim):
#         super(ServerModel, self).__init__()
#         self.encoder = ResNet18Enc(z_dim=z_dim)
#         self.decoder = ResNet18Dec(z_dim=z_dim)

#     def forward(self, x):
#         mean, logvar = self.encoder(x)
#         z = self.reparameterize(mean, logvar)
#         x = self.decoder(z)
#         return x, mean, logvar

#     @staticmethod
#     def reparameterize(mean, logvar):
#         std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
#         epsilon = torch.randn_like(std).cuda()
#         return epsilon * std + mean


# def loss_func(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD



# In this experiment, smaller loss doesn't mean better generative capacity.
# In fact, no convergence model has a better generative ability. (see the generated image)
# So We could train the VAE just in a few epoches?
class ServerModel(nn.Module):
    def __init__(self, num_classes):
        super(ServerModel, self).__init__()

        label_embedding_size = 512
        classifier_features = args.latent_size
        self.VAE_latent_size = args.latent_size
        # z_dim = args.VAE_size

        self.label_embedding = nn.Embedding(num_classes, label_embedding_size)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=args.latent_size + label_embedding_size,
                      out_features=classifier_features),
            nn.BatchNorm1d(classifier_features),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # nn.BatchNorm1d(classifier_features),
            # nn.ReLU(),

            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # nn.BatchNorm1d(classifier_features),
            # nn.ReLU(),

            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # nn.BatchNorm1d(classifier_features),
            # nn.ReLU(),
            
            # nn.Sigmoid(),
            nn.Linear(in_features=classifier_features, 
                      out_features=self.VAE_latent_size * 2)
        )

        

        # self.mean_fc = nn.Linear(in_features=classifier_features,
        #                          out_features=self.VAE_latent_size)

        # self.var_fc = nn.Linear(in_features=classifier_features,
        #                         out_features=self.VAE_latent_size)


        # self.sixth_layer = nn.Flatten()
        

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.VAE_latent_size + label_embedding_size,
                      out_features=args.latent_size),
            # nn.BatchNorm1d(args.latent_size),
            # nn.LeakyReLU(0.2),
            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # nn.BatchNorm1d(classifier_features),
            # nn.Sigmoid(),
            nn.Linear(in_features=args.latent_size, 
                      out_features=args.latent_size),
            # nn.Sigmoid()
        )
        
        # self.classifier = ResNet18Dec(z_dim=z_dim)

        # self.seventh_layer = nn.Linear(in_features=layer2*self.pool2_dim*self.pool2_dim, 
        #                                out_features=num_classes, 
        #                                bias=False)



    def encode(self, x, labels):
        labels = self.label_embedding(labels)
        concat_vec = torch.cat([x, labels], dim=1)
        concat_vec = self.feature_extractor(concat_vec)
        mean = concat_vec[:, :self.VAE_latent_size]
        var = concat_vec[:, self.VAE_latent_size:]
        # return self.mean_fc(concat_vec), self.var_fc(concat_vec), labels
        return mean, var, labels

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        return self.classifier(z)

    def forward(self, x, labels):
        mean, var, labels = self.encode(x, labels)
        z = self.reparametrize(mean, var)
        return self.decode(z, labels), mean, var

    @torch.no_grad()
    def generate_data(self, z, labels):
        labels = self.label_embedding(labels)
        return self.decode(z, labels)






def server_extractor_training(extractor, 
                              num_epochs,
                              dataloader,
                              **kwargs):

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    # running_device = kwargs["device"]
    # batch_size = kwargs["batch_size"]
    # latent_size = kwargs["latent_size"]

    model = extractor

    reconstruction_function = nn.MSELoss()


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
            x_data, y_label = data
            x_data = x_data.view(x_data.size(0), -1)
            # if torch.cuda.is_available():
            #     x_data = x_data.cuda()
            #     y_label = y_label.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_data, y_label)
            loss = loss_function(recon_batch, x_data, mu, logvar)
            loss.backward()
            
            # nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)
            
            train_loss += loss.item()
            optimizer.step()
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * len(x_data),
            #         len(dataloader.dataset), 100. * batch_idx / len(dataloader),
            #         loss.item() / len(img)))
        # if epoch == num_epochs - 1:
        #     print('====> Epoch: {} Average loss: {:.4f}'.format(
        #         epoch, train_loss / len(dataloader.dataset)))
        # if epoch % 10 == 0:
        #     save = to_img(recon_batch.cpu().data)
        #     save_image(save, './vae_img/image_{}.png'.format(epoch))


        # if epoch <= 20 or epoch % 10 == 0:
        #     with torch.no_grad():
        #         for i in range(10):
        #             eps = torch.randn((batch_size, latent_size)).cuda()
        #             fake_label = torch.full((batch_size, ), fill_value=i, dtype=torch.int32).cuda()
        #             generate_data = model.generate_data(eps, fake_label)
        #             save = to_img(generate_data.cpu().data)
        #             if not os.path.exists('./vae_img/{}/'.format(i)):
        #                 os.mkdir('./vae_img/{}/'.format(i))
        #             save_image(save, './vae_img/{}/generated_image_{}.png'.format(i, epoch))
    return model
    # torch.save(model.state_dict(), './vae.pth')



    # eps = torch.randn((batch_size, latent_size)).cuda()
    # fake_label = torch.ones((batch_size), dtype=torch.int32).cuda()
    # generate_img = model.generate_img(eps, fake_label)
    # save = to_img(generate_img.cpu().data)
    # save_image(save, './vae_img/generated_image_{}.png'.format(1))
