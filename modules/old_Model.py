
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#### Create tf model for Client ####

class ClientModel(nn.Module):
    def __init__(self, input_shape, num_classes, layer1=128, layer2=192, layer3=256, dropout_rate=0.2):
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
        self.conv1 = nn.Conv2d(in_channels=input_shape[1] , out_channels=layer1, 
                               kernel_size=(3, 3), padding='same')

        self.conv1_dim = self.conv_output_dimension(input_shape[3], 
                                                    kernel=3, 
                                                    padding=1, 
                                                    stride=1)
        
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(layer1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # input size: [n, layer1, 32, 32]
        # output size: [n, ]
        self.pooling1 = nn.AvgPool2d((2, 2), stride=1, padding=1)
        self.pool1_dim = self.avgpool_output_dimension(self.conv1_dim, 
                                                       kernel=2, 
                                                       padding=1, 
                                                       stride=1)
        
        # 2nd convolution layer

        self.conv2 = nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=(3, 3), stride=2,
                               padding='valid')
        self.conv2_dim = self.conv_output_dimension(self.pool1_dim, 
                                                    kernel=3,
                                                    padding=0,
                                                    stride=2)


        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(layer2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.pooling2 = nn.AvgPool2d((2, 2), stride=2, padding=0)
        self.pool2_dim = self.avgpool_output_dimension(self.conv2_dim, 
                                                       kernel=2, 
                                                       padding=0, 
                                                       stride=2)
        # 3rd convolution layer

        self.conv3 = nn.Conv2d(in_channels=layer2, out_channels=layer3, kernel_size=(3, 3), stride=2,
                               padding='valid')

        self.conv3_dim = self.conv_output_dimension(self.pool2_dim, 
                                                    kernel=3,
                                                    padding=0,
                                                    stride=2)


        self.bn3 = nn.Sequential(
            nn.BatchNorm2d(layer3),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )


        # self.sixth_layer = nn.Flatten()

        
        self.seventh_layer = nn.Linear(in_features=layer3*self.conv3_dim*self.conv3_dim, 
                                       out_features=num_classes, 
                                       bias=False)



    def forward(self, inputs):
        x = self.conv1(inputs)
        # if training:
        x = self.bn1(x)
            # print("in_training")
        x = self.pooling1(x)

        x = self.conv2(x)
        # if training:
        x = self.bn2(x)
        x = self.pooling2(x)

        x = self.conv3(x)
        # if training:
        x = self.bn3(x)

        x = x.view(x.shape[0], -1)
        x = self.seventh_layer(x)
        return x

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

        self.conv1 = nn.Conv2d(in_channels=input_shape[1] , out_channels=layer1, 
                               kernel_size=(3, 3), padding='same')

        self.conv1_dim = self.conv_output_dimension(input_shape[3], 
                                                    kernel=3, 
                                                    padding=1, 
                                                    stride=1)

        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(layer1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            
        )
        
        self.pooling1 = nn.AvgPool2d((2, 2), stride=1, padding=1)
        self.pool1_dim = self.avgpool_output_dimension(self.conv1_dim, 
                                                       kernel=2, 
                                                       padding=1, 
                                                       stride=1)
        
        # 2nd convolution layer

        self.conv2 = nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=(3, 3), stride=2,
                               padding='valid')

        self.conv2_dim = self.conv_output_dimension(self.pool1_dim, 
                                                    kernel=3,
                                                    padding=0,
                                                    stride=2)

        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(layer2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.pooling2 = nn.AvgPool2d((2, 2), stride=2, padding=0)

        self.pool2_dim = self.avgpool_output_dimension(self.conv2_dim, 
                                                       kernel=2, 
                                                       padding=0, 
                                                       stride=2)

        self.sixth_layer = nn.Flatten()

        self.seventh_layer = nn.Linear(in_features=layer2*self.pool2_dim*self.pool2_dim, 
                                       out_features=num_classes, 
                                       bias=False)



    def forward(self, inputs):
        x = self.conv1(inputs)
        # if training:
        x = self.bn1(x)
            # print("in_training")
        x = self.pooling1(x)

        x = self.conv2(x)
        # if training:
        x = self.bn2(x)
        x = self.pooling2(x)

        x = self.sixth_layer(x)
        x = self.seventh_layer(x)
        return x

    def conv_output_dimension(self, input_height, kernel, padding, stride=1, dilation=1):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - dilation * (kernel - 1) - 1)/stride)
        return int(height_out)

    def avgpool_output_dimension(self, input_height, kernel, padding, stride):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - kernel)/stride)
        return int(height_out)


