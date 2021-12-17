import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.modules.linear import Linear
from tqdm import tqdm

import matplotlib.pyplot as plt
import os


from modules.utils import get_model_mask
from parameters import args



def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)




# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, n_class, features_base=256, batch_size=64):
        super(Generator, self).__init__()
        self.is_bias = False
        self.n_classes = n_class
        self.batch_size = batch_size
        # self.label_embedding_generator = nn.Sequential(
        #     # nn.Embedding(n_class, 5 * n_class), 
        #     nn.Linear(n_class, 5 * n_class)
        # )

        # self.noise_latent_gen = nn.Sequential(
        #     nn.Linear(nz, 5 * nz),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )




        self.main = nn.Sequential(
            # input is Z, going into a Linear (Regression)
            nn.Conv1d(in_channels=nz + n_class, out_channels=features_base * 1, kernel_size=1),
            nn.BatchNorm1d(features_base * 1),
            nn.LeakyReLU(True),
            # nn.MaxPool1d(3, stride=2, padding=1),
            # state size. features_base
            nn.Conv1d(in_channels=features_base * 1, out_channels=features_base * 2, kernel_size=1),
            nn.BatchNorm1d(features_base * 2),
            nn.LeakyReLU(True),
            # nn.AvgPool2d((2, 2), stride=2, padding=0),
            # state size. features_base * 4
            # nn.Linear(features_base * 4, features_base * 8, bias=self.is_bias),
            # nn.BatchNorm1d(features_base * 8),
            # nn.ReLU(True),
            # # state size. features_base * 8
            # nn.Linear(features_base * 8, features_base * 4, bias=self.is_bias),
            # nn.BatchNorm1d(features_base * 4),
            # nn.ReLU(True),
            # state size. features_base * 4
            nn.Conv1d(in_channels=features_base * 2, out_channels=features_base * 1, kernel_size=1),
            nn.BatchNorm1d(features_base * 1),
            nn.LeakyReLU(True),
            # nn.AvgPool2d((2, 2), stride=1, padding=1),
            # state size. features_base * 2
            # nn.Linear(features_base * 2, features_base, bias=self.is_bias),
            # nn.BatchNorm1d(features_base),
            # nn.ReLU(True),
            nn.Flatten(),
            # state size. features_base
            nn.Linear(features_base, n_class, bias=self.is_bias),
            # nn.Tanh()
        )


    def forward(self, noises, labels):
        # labels = self.label_embedding_generator(labels)
        labels = F.one_hot(labels, self.n_classes)
        # noises_output = self.noise_latent_gen(noises)
        # labels = labels.view(-1, 1)
        concat_vector = torch.cat([noises, labels], dim=1)
        concat_vector = torch.unsqueeze(concat_vector, 2)
        # concat_vector.requires_grad_()
        return self.main(concat_vector)

    def conv1d_output_dimension(self, input_len, kernel, padding, stride=1, dilation=1):
        len_in = input_len
        len_out = np.floor(1 + (len_in + 2 * padding - dilation * (kernel - 1) - 1)/stride)
        return int(len_out)

    # def avgpool_output_dimension(self, input_height, kernel, padding, stride):
    #     height_in = input_height
    #     height_out = np.floor(1 + (height_in + 2 * padding - kernel)/stride)
    #     return int(height_out)


class Discriminator(nn.Module):
    def __init__(self, nz, n_class, features_base=256):
        super(Discriminator, self).__init__()
        self.is_bias = False
        self.n_classes = n_class

        # self.label_embedding_discriminator = nn.Sequential(
        #     # nn.Embedding(n_class, 5 * n_class), 
        #     nn.Linear(n_class, 5 * n_class)
        # )

        # self.noise_latent_disc = nn.Sequential(
        #     nn.Linear(nz, 10 * nz),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )


        self.main = nn.Sequential(
            # input is Z, going into a Linear (Regression)
            nn.Conv1d(in_channels=nz + n_class, out_channels=features_base * 1, kernel_size=1),
            nn.BatchNorm1d(features_base * 1),
            nn.LeakyReLU(True),
            # nn.MaxPool1d(3, stride=2, padding=1),
            # state size. features_base
            nn.Conv1d(in_channels=features_base * 1, out_channels=features_base * 2, kernel_size=1),
            nn.BatchNorm1d(features_base * 2),
            nn.LeakyReLU(True),
            # nn.AvgPool2d((2, 2), stride=2, padding=0),
            # state size. features_base * 4
            # nn.Linear(features_base * 4, features_base * 8, bias=self.is_bias),
            # nn.BatchNorm1d(features_base * 8),
            # nn.ReLU(True),
            # # state size. features_base * 8
            # nn.Linear(features_base * 8, features_base * 4, bias=self.is_bias),
            # nn.BatchNorm1d(features_base * 4),
            # nn.ReLU(True),
            # state size. features_base * 4
            nn.Conv1d(in_channels=features_base * 2, out_channels=features_base * 1, kernel_size=1),
            nn.BatchNorm1d(features_base * 1),
            nn.LeakyReLU(True),
            # state size. features_base * 2
            # nn.Linear(features_base * 2, features_base, bias=self.is_bias),
            # nn.BatchNorm1d(features_base),
            # nn.ReLU(True),
            nn.Flatten(),
            # state size. features_base
            nn.Linear(features_base, 1, bias=self.is_bias),
            # nn.Sigmoid()
        )

    def forward(self, noises, labels):
        # labels = self.label_embedding_discriminator(labels)
        # noise_output = self.noise_latent_disc(noises)
        labels = F.one_hot(labels, self.n_classes)
        # labels = labels.view(-1, 1)
        concat_vector = torch.cat([noises, labels], dim=1)
        concat_vector = torch.unsqueeze(concat_vector, 2)
        output = self.main(concat_vector)
        if not args.gradient_penalty:
            output = torch.sigmoid(output)
        return output

    def conv1d_output_dimension(self, input_len, kernel, padding, stride=1, dilation=1):
        len_in = input_len
        len_out = np.floor(1 + (len_in + 2 * padding - dilation * (kernel - 1) - 1)/stride)
        return int(len_out)

    # def avgpool_output_dimension(self, input_height, kernel, padding, stride):
    #     height_in = input_height
    #     height_out = np.floor(1 + (height_in + 2 * padding - kernel)/stride)
    #     return int(height_out)



# def get_generator_noise(batch_size, noise_size, num_classes, device, y_list=None):
#     noise = torch.randn(batch_size, noise_size, device=device)
#     # onehot_label_noise = F.one_hot(y_list, num_classes)
#     # x_y_noise = torch.cat([noise, onehot_label_noise], dim=1)
#     return noise

def calc_gradient_penalty(netD, real_data, fake_data, x_label, device):
    # print "real_data: ", real_data.size(), fake_data.size()
    LAMBDA = 10 # Gradient penalty lambda hyperparameter
    real_data_shape = real_data.size()
    batch_size = real_data_shape[0]
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, real_data_shape[1]).contiguous()
    # alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # if use_cuda:
        # interpolates = interpolates.cuda(gpu)
    # interpolates = torch.Tensor(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, x_label)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def training_loop(generator, 
                  discriminator, 
                  noise_size, 
                  num_classes,
                  num_epochs, 
                  device,
                  dataloader):
    criterion = nn.BCELoss()
    G_losses = []
    D_losses = []
    iters = 0
    # Learning rate for optimizers
    lr = 0.0001
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Setup Adam optimizers for both G and D
    if args.GAN_moment_flag:
        optimizerD = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        optimizerG = torch.optim.RMSprop(generator.parameters(), lr=lr)
    else:
        optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    # optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)
    # optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    errG, errD = 0, 0


    for epoch in tqdm(range(num_epochs)):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            if i == 300:
                break
            ############################
            # (1) Update D network: maximize log(D(x|y)) + log(1 - D(G(z|y)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_x = data[0]
            real_x_label = data[1]
            b_size = real_x.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_x, real_x_label).view(-1)
            # Calculate loss on all-real batch
            if not args.gradient_penalty:
                errD_real = criterion(output, label)
            else:
                errD_real = output.mean()
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, noise_size, device=device)
            # Generate fake data batch with G
            fake = generator(noise, real_x_label)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach(), real_x_label).view(-1)
            # Calculate D's loss on the all-fake batch
            if not args.gradient_penalty:
                errD_fake = criterion(output, label)
            else:
                errD_fake = output.mean()
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches

            if args.gradient_penalty:
                gradient_penalty = calc_gradient_penalty(discriminator, 
                                                        real_x, fake, 
                                                        real_x_label, 
                                                        device)
                gradient_penalty.backward()


            errD = errD_fake + errD_real
            # Update D
            optimizerD.step()


            ###########################
            # (2) Update G network: maximize log(D(G(z|y)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Generate fake data batch with G
            fake = generator(noise, real_x_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake, real_x_label).view(-1)
            # Calculate G's loss based on this output
            if not args.gradient_penalty:
                errG = criterion(output, label)
            else:
                errG = output.mean()
            # print(generator)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            iters += 1

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())


        # Output training stats
        if epoch % int(num_epochs * 0.1) == 0 and epoch > 0:
            print('[{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f} / {:.4f}'.format
                (epoch, num_epochs,
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            
            # G_losses = errG.item()
            # D_losses = errD.item()

        # Check how the generator is doing by saving G's output on fixed_noise
        if (epoch % int(num_epochs * 0.1) == 0):
            # check the generated sample during the training
            check_size = 5
            # select the same label for showing a better picture
            select_label_idx = torch.randint(0, 10, (1, ))
            label_noise = torch.randint(int(select_label_idx.numpy()), int(select_label_idx.numpy()) + 1, (check_size, ), device=device)
            # get noise inputs
            noise = torch.randn(check_size, num_classes, device=device)
            with torch.no_grad():
                fake = generator(noise, label_noise).detach().cpu()
            fake = fake.numpy()
            label_noise = label_noise.cpu().numpy()
            for idx in range(fake.shape[0]):
                plt.plot(fake[idx, :], label=str(label_noise[idx]))
            plt.title("Generated samples")
            plt.legend()
            model_mask = get_model_mask(args)
            pic_save_path = "pic/{}/{}/alpha{}_lr{}/".format(args.dataset, model_mask, args.alpha, args.learning_rate)
            makedirs(pic_save_path)
            plt.savefig("{}Generated_samples_training{}.png".format(pic_save_path, epoch/num_epochs * 100))
            plt.close()


    return G_losses, D_losses


