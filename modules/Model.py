
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.modules.activation import ReLU

from parameters import args 
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#### Create tf model for Client ####


def conv_shortcut(in_channel,out_channel,stride):
    layers = [nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),stride=(stride,stride)),
             nn.BatchNorm2d(out_channel)]
    return nn.Sequential(*layers)

def block(in_channel,out_channel,k_size,stride, conv=False):
    layers = None
    
    first_layers = [nn.Conv2d(in_channel,out_channel[0],kernel_size=(1,1),stride=(1,1)),
                    nn.BatchNorm2d(out_channel[0]),
                    nn.ReLU(inplace=True)]
    if conv:
        first_layers[0].stride=(stride,stride)
    
    second_layers = [nn.Conv2d(out_channel[0],out_channel[1],kernel_size=(k_size,k_size),stride=(1,1),padding=1),
                    nn.BatchNorm2d(out_channel[1])]

    layers = first_layers + second_layers
    
    return nn.Sequential(*layers)

class ClientModel(nn.Module):
    def __init__(self, input_shape, num_classes, layer1=64, layer2=256, layer3=256, dropout_rate=0.2):
        super(ClientModel, self).__init__()

        # The output size of convolution layer: [(W-K+2P)/S]+1
        # The output size of maxpooling layer: (W-K)/S + 1
        # W is the input volume: image is 32*32*3, W=32
        # K is the kernel size
        # P is the padding
        # S is the stride

        # 1st convolution layer
        # input size: [n, 3, 32, 32]
        # output size: [n, layer1, 32, 32]

        label_embedding_size = 256
        classifier_features = 256

        self.conv1 = nn.Conv2d(in_channels=input_shape[1] , out_channels=layer1, 
                               kernel_size=(3), stride=(1), padding=1)
        
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(layer1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.conv2 = block(64,[64,256],3,1,conv=True)
        self.ident2 = block(256,[64,256],3,1)
        self.conv3 = block(256,[128,512],3,2,conv=True)
        self.ident3 = block(512,[128,512],3,2)
        self.conv4 = block(512,[256,1024],3,2,conv=True)
        # self.ident4 = block(1024,[256,1024],3,2)
        # self.conv2_dim = self.conv_output_dimension(self.pool1_dim, 
        #                                             kernel=3,
        #                                             padding=0,
        #                                             stride=2)


        # self.bn2 = nn.Sequential(
        #     nn.BatchNorm2d(layer2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate)
        # )
        # self.pooling2 = nn.AvgPool2d((2, 2), stride=2, padding=0)
        # self.pool2_dim = self.avgpool_output_dimension(self.conv2_dim, 
        #                                                kernel=2, 
        #                                                padding=0, 
        #                                                stride=2)
        # # 3rd convolution layer
        # self.conv3 = nn.Conv2d(in_channels=layer2, out_channels=layer3, kernel_size=(3, 3), stride=2,
        #                        padding='valid')

        # self.conv3_dim = self.conv_output_dimension(self.pool2_dim, 
        #                                             kernel=3,
        #                                             padding=0,
        #                                             stride=2)


        # self.bn3 = nn.Sequential(
        #     nn.BatchNorm2d(layer3),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate)
        # )

        # self.label_embedding = nn.Embedding(num_classes, label_embedding_size)

        self.flatten = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4)),
            nn.Flatten(),
            # nn.Linear(1024, args.latent_size)
        )

        # self.var_fc = nn.Sequential(
        #     nn.Linear(in_features=layer3*self.conv3_dim*self.conv3_dim,
        #                 out_features=args.latent_size),
        #     nn.BatchNorm1d(args.latent_size),
        #     nn.ReLU(),
        #     nn.Linear(in_features=args.latent_size, out_features=args.latent_size)
        # )

        # self.mean_fc = nn.Linear(in_features=layer3*self.conv3_dim*self.conv3_dim,
        #                          out_features=args.latent_size)

        # self.var_fc = nn.Linear(in_features=layer3*self.conv3_dim*self.conv3_dim,
        #                         out_features=args.latent_size)



        # self.sixth_layer = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=args.latent_size,
                      out_features=num_classes),
            # # nn.BatchNorm1d(classifier_features),
            # # nn.ReLU(),
            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # # nn.BatchNorm1d(classifier_features),
            # # nn.ReLU(),
            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # # nn.BatchNorm1d(classifier_features),
            # # nn.ReLU(),
            # nn.Linear(in_features=classifier_features, 
            #           out_features=num_classes, 
            #           bias=False)
        )



        # self.seventh_layer = nn.Linear(in_features=layer3*self.conv3_dim*self.conv3_dim, 
        #                                out_features=num_classes, 
        #                                bias=False)


    def encode(self, x):
        # labels = self.label_embedding(labels)

        x = self.conv1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.ident2(x))
        x = F.relu(self.ident2(x))
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.ident3(x))
        x = F.relu(self.ident3(x))
        
        x = F.relu(self.conv4(x))
        # x = F.relu(self.ident4(x))
        # x = F.relu(self.ident4(x))

        # x = x.view(x.shape[0], -1)

        # concat_vec = torch.cat([x, labels], dim=1)
        # concat_vec = self.fc_label_img(concat_vec)
        return self.flatten(x)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if torch.cuda.is_available():
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     return eps.mul(std).add_(mu)

    def decode(self, z):
        # z = torch.cat([z, labels], dim=1)
        return self.classifier(z)

    def forward(self, x):
        z = self.encode(x)
        # z = self.reparametrize(mean, logvar)
        return self.decode(z)

    def conv_output_dimension(self, input_height, kernel, padding, stride=1, dilation=1):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - dilation * (kernel - 1) - 1)/stride)
        return int(height_out)

    def avgpool_output_dimension(self, input_height, kernel, padding, stride):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - kernel)/stride)
        return int(height_out)


class ClientModel_2CNN(nn.Module):
    def __init__(self, input_shape, num_classes, layer1=128, layer2=192, layer3=256, dropout_rate=0.2):
        super(ClientModel_2CNN, self).__init__()

        # The output size of convolution layer: [(W-K+2P)/S]+1
        # The output size of maxpooling layer: (W-K)/S + 1
        # W is the input volume: image is 32*32*3, W=32
        # K is the kernel size
        # P is the padding
        # S is the stride

        # 1st convolution layer

        label_embedding_size = 256
        classifier_features = 256

        self.conv1 = nn.Conv2d(in_channels=input_shape[1] , out_channels=layer1, 
                               kernel_size=(3), stride=(1), padding=1)
        
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(layer1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.conv2 = block(64,[64,256],3,1,conv=True)
        self.ident2 = block(256,[64,256],3,1)
        self.conv3 = block(256,[128,512],3,2,conv=True)
        self.ident3 = block(512,[128,512],3,2)
        self.conv4 = block(512,[256,1024],3,2,conv=True)
        # self.ident4 = block(1024,[256,1024],3,2)

        self.flatten = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4)),
            nn.Flatten(),
            # nn.Linear(1024, args.latent_size)
        )
        

        # self.label_embedding = nn.Embedding(num_classes, label_embedding_size)

        # self.var_fc = nn.Sequential(
        #     nn.Linear(in_features=layer2*self.conv2_dim*self.conv2_dim,
        #                 out_features=args.latent_size),
        #     nn.BatchNorm1d(args.latent_size),
        #     nn.ReLU(),
        #     nn.Linear(in_features=args.latent_size, out_features=args.latent_size)
        # )

        # self.mean_fc = nn.Linear(in_features=layer2*self.conv2_dim*self.conv2_dim,
        #                          out_features=args.latent_size)

        # self.var_fc = nn.Linear(in_features=layer2*self.conv2_dim*self.conv2_dim,
        #                         out_features=args.latent_size)


        # self.sixth_layer = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=args.latent_size,
                      out_features=num_classes),
            # nn.BatchNorm1d(classifier_features),
            # nn.ReLU(),
            # nn.Linear(in_features=classifier_features,
            #           out_features=classifier_features),
            # # nn.BatchNorm1d(classifier_features),
            # # nn.ReLU(),
            # nn.Linear(in_features=classifier_features, 
            #           out_features=num_classes, 
            #           bias=False)
        )

        # self.seventh_layer = nn.Linear(in_features=layer2*self.pool2_dim*self.pool2_dim, 
        #                                out_features=num_classes, 
        #                                bias=False)



    def encode(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.ident2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.ident3(x))
        x = F.relu(self.conv4(x))
        # x = F.relu(self.ident4(x))
        # x = x.view(x.shape[0], -1)

        # concat_vec = torch.cat([x, labels], dim=1)
        # concat_vec = self.fc_label_img(concat_vec)
        return self.flatten(x)


    def decode(self, z):
        # z = torch.cat([z], dim=1)
        return self.classifier(z)

    def forward(self, x):
        z = self.encode(x)
        # z = self.reparametrize(mean, logvar)
        return self.decode(z)

    def conv_output_dimension(self, input_height, kernel, padding, stride=1, dilation=1):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - dilation * (kernel - 1) - 1)/stride)
        return int(height_out)

    def avgpool_output_dimension(self, input_height, kernel, padding, stride):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - kernel)/stride)
        return int(height_out)
        

