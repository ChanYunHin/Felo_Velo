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
            
            nn.Linear(in_features=classifier_features, 
                      out_features=self.VAE_latent_size * 2)
        )

        

        

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.VAE_latent_size + label_embedding_size,
                      out_features=args.latent_size),
            
            nn.Linear(in_features=args.latent_size, 
                      out_features=args.latent_size),
        )
        



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
            
    return model
