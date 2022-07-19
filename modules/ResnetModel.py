
from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import args

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


# class Resnet34(ResNet):
    
#     def __init__(self, num_classes=10):
#         super(Resnet34, self).__init__(BasicBlock, [3, 4, 6, 3])
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         self.num_feature = self.fc.in_features
#         self.fc = nn.Linear(self.num_feature, num_classes)

#     def forward(self, x):
        
#         x = self.encode(x)
#         x = self.decode(x)
        
#         return x

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        
#         return x

#     def decode(self, z):
#         # z = torch.cat([z, labels], dim=1)
#         return self.fc(z)


# class Resnet18(ResNet):
    
#     def __init__(self, num_classes=10):
#         super(Resnet18, self).__init__(BasicBlock, [2, 2, 2, 2])
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         self.num_feature = self.fc.in_features
#         self.fc = nn.Linear(self.num_feature, num_classes)

#     def forward(self, x):
        
#         x = self.encode(x)
#         x = self.decode(x)
        
#         return x

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        
#         return x

#     def decode(self, z):
#         # z = torch.cat([z, labels], dim=1)
#         return self.fc(z)

# class Resnet22(ResNet):
    
#     def __init__(self, num_classes=10):
#         super(Resnet22, self).__init__(BasicBlock, [2, 3, 3, 2])
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         self.num_feature = self.fc.in_features
#         self.fc = nn.Linear(self.num_feature, num_classes)

#     def forward(self, x):
        
#         x = self.encode(x)
#         x = self.decode(x)
        
#         return x

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        
#         return x

#     def decode(self, z):
#         # z = torch.cat([z, labels], dim=1)
#         return self.fc(z)

# class Resnet26(ResNet):
    
#     def __init__(self, num_classes=10):
#         super(Resnet26, self).__init__(BasicBlock, [3, 3, 3, 3])
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         self.num_feature = self.fc.in_features
#         self.fc = nn.Linear(self.num_feature, num_classes)

#     def forward(self, x):
        
#         x = self.encode(x)
#         x = self.decode(x)
        
#         return x

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        
#         return x

#     def decode(self, z):
#         # z = torch.cat([z, labels], dim=1)
#         return self.fc(z)
    
# class Resnet30(ResNet):
    
#     def __init__(self, num_classes=10):
#         super(Resnet30, self).__init__(BasicBlock, [3, 4, 4, 3])
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         self.num_feature = self.fc.in_features
#         self.fc = nn.Linear(self.num_feature, num_classes)

#     def forward(self, x):
        
#         x = self.encode(x)
#         x = self.decode(x)
        
#         return x

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        
#         return x

#     def decode(self, z):
#         # z = torch.cat([z, labels], dim=1)
#         return self.fc(z)



# class Resnet34_ft(Resnet34):
#     def __init__(self, num_classes=10):
#         super(Resnet34_ft, self).__init__(num_classes)
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         # self.num_feature = self.fc.in_features
#         # self.fc = nn.Linear(self.num_feature, num_classes)

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.maxpool(x)

#         # x = self.layer1(x)
#         # x = self.layer2(x)
#         # x = self.layer3(x)
#         # x = self.layer4(x)

#         return x

#     def decode(self, x):
#         # x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)


# class Resnet18_ft(Resnet18):
#     def __init__(self, num_classes=10):
#         super(Resnet18_ft, self).__init__(num_classes)
#         # self.z_dim = z_dim
#         # self.ResNet18 = models.resnet18(pretrained=False)
#         # self.num_feature = self.fc.in_features
#         # self.fc = nn.Linear(self.num_feature, num_classes)

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.maxpool(x)

#         # x = self.layer1(x)
#         # x = self.layer2(x)
#         # x = self.layer3(x)
#         # x = self.layer4(x)

#         return x

#     def decode(self, x):
#         # x = self.conv1(x)
#         # x = self.bn1(x)
#         # x = self.relu(x)
#         # x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)

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
    

class Resnet50(nn.Module):
    
    def __init__(self,in_channels,num_classes):
        super().__init__()
        
        self.stg1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        
        ##stage 2
        self.convShortcut2 = conv_shortcut(64,256,1)
        
        self.conv2 = block(64,[64,256],3,1,conv=True)
        self.ident2 = block(256,[64,256],3,1)

        
        ##stage 3
        self.convShortcut3 = conv_shortcut(256,512,2)
        
        self.conv3 = block(256,[128,512],3,2,conv=True)
        self.ident3 = block(512,[128,512],3,2)

        
        ##stage 4
        self.convShortcut4 = conv_shortcut(512,1024,2)
        
        self.conv4 = block(512,[256,1024],3,2,conv=True)
        self.ident4 = block(1024,[256,1024],3,2)
        
        self.tmp_layer = nn.Sequential(
            # nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(args.avgpool)),
            nn.Flatten(),
            # nn.Linear(1024, args.latent_size)
        )
        

        classifier_features = 256


        if args.one_classifer_flag:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=args.latent_size, out_features=num_classes),
            )
        else:
            ##Classify
            self.classifier = nn.Sequential(
                nn.Linear(in_features=args.latent_size, out_features=classifier_features),
                # nn.BatchNorm1d(classifier_features),
                # nn.ReLU(),

                nn.Linear(in_features=classifier_features,
                        out_features=classifier_features),

                nn.Linear(in_features=classifier_features,
                        out_features=classifier_features),
        
                # nn.BatchNorm1d(classifier_features),
                # nn.ReLU(),
                nn.Linear(in_features=classifier_features, 
                        out_features=num_classes)
            )
        
    def forward(self,inputs):
        
        out = self.encode(inputs)
        
        out = self.decode(out)
        
        # out = self.stg1(inputs)
        
        # #stage 2
        # out = F.relu(self.conv2(out) + self.convShortcut2(out))
        # out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        # #stage3
        # out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        # out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # #stage4            
        # out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        # out = F.relu(self.ident4(out) + out)
        # out = F.relu(self.ident4(out) + out)
        # out = F.relu(self.ident4(out) + out)
        # out = F.relu(self.ident4(out) + out)
        # out = F.relu(self.ident4(out) + out)

        # out = self.tmp_layer(out)
        
        # out = self.classifier(out)#100x1024
        
        return out

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        
        #stage4             
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)

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

class Resnet50_ft(Resnet50):
    
    def __init__(self,in_channels,num_classes):
        super(Resnet50_ft, self).__init__(in_channels, num_classes)
        
        # self.stg1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
        #                           nn.BatchNorm2d(64),
        #                           nn.ReLU(inplace=True))
        
    def forward(self,inputs):
        
        encoded_feature_inputs = self.encode(inputs)
        
        encoded_features = self.decode(encoded_feature_inputs)
        
        # return self.classifier(encoded_features), encoded_feature_inputs, encoded_features
        return self.classifier(encoded_features)

    def encode(self, inputs):
        out = self.stg1(inputs)
        return out

    def decode(self, out):
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        
        #stage4             
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        out = self.tmp_layer(out)
        
        
        return out
    
    def fc_layers(self, out):
        return self.classifier(out)


class SmallResnet50(nn.Module):
    
    def __init__(self,in_channels,num_classes):
        super().__init__()
        
        self.stg1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        
        ##stage 2
        self.convShortcut2 = conv_shortcut(64,256,1)
        
        self.conv2 = block(64,[64,256],3,1,conv=True)
        self.ident2 = block(256,[64,256],3,1)

        
        ##stage 3
        self.convShortcut3 = conv_shortcut(256,512,2)
        
        self.conv3 = block(256,[128,512],3,2,conv=True)
        self.ident3 = block(512,[128,512],3,2)
        
        ##stage 4
        self.convShortcut4 = conv_shortcut(512,1024,2)
        
        self.conv4 = block(512,[256,1024],3,2,conv=True)
        self.ident4 = block(1024,[256,1024],3,2)

        
        self.tmp_layer = nn.Sequential(
            # nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(args.avgpool)),
            nn.Flatten(),
            # nn.Linear(1024, args.latent_size)
        )
        

        classifier_features = 256


        
        if args.one_classifer_flag:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=args.latent_size, out_features=num_classes),
            )
        else:
            ##Classify
            self.classifier = nn.Sequential(
                nn.Linear(in_features=args.latent_size, out_features=classifier_features),
                # nn.BatchNorm1d(classifier_features),
                # nn.ReLU(),

                nn.Linear(in_features=classifier_features,
                        out_features=classifier_features),

                nn.Linear(in_features=classifier_features,
                        out_features=classifier_features),
        
                # nn.BatchNorm1d(classifier_features),
                # nn.ReLU(),
                nn.Linear(in_features=classifier_features, 
                        out_features=num_classes)
            )
        
    def forward(self,inputs):
        
        out = self.encode(inputs)
        
        out = self.decode(out)
        
        return out

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)


    def decode(self, z):
        # z = torch.cat([z, labels], dim=1)
        return self.classifier(z)


class SmallResnet1(SmallResnet50):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
    
    def forward(self,inputs):
        
        out = self.encode(inputs)
        
        out = self.decode(out)
        
        return out

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)


    def decode(self, z):
        # z = torch.cat([z, labels], dim=1)
        return self.classifier(z)
    
class SmallResnet1_ft(SmallResnet1):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        
    def forward(self,inputs):
        
        encoded_feature_inputs = self.encode(inputs)
        
        encoded_features = self.decode(encoded_feature_inputs)
        
        # return self.classifier(encoded_features), encoded_feature_inputs, encoded_features
        return self.classifier(encoded_features)

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        return out


    def decode(self, out):
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)
        
        # return out
    
    def fc_layers(self, out):
        return self.classifier(out)

    
class SmallResnet2(SmallResnet50):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
    
    def forward(self,inputs):
        
        out = self.encode(inputs)
        
        out = self.decode(out)
        
        return out

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)


    def decode(self, z):
        # z = torch.cat([z, labels], dim=1)
        return self.classifier(z)
    
class SmallResnet2_ft(SmallResnet2):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        
    def forward(self,inputs):
        
        encoded_feature_inputs = self.encode(inputs)
        
        encoded_features = self.decode(encoded_feature_inputs)
        
        # return self.classifier(encoded_features), encoded_feature_inputs, encoded_features
        return self.classifier(encoded_features)

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        return out


    def decode(self, out):
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)
        
        # return out
    
    def fc_layers(self, out):
        return self.classifier(out)

class SmallResnet3(SmallResnet50):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
    
    def forward(self,inputs):
        
        out = self.encode(inputs)
        
        out = self.decode(out)
        
        return out

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        # out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)


    def decode(self, z):
        # z = torch.cat([z, labels], dim=1)
        return self.classifier(z)
    

class SmallResnet3_ft(SmallResnet3):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        
    def forward(self,inputs):
        
        encoded_feature_inputs = self.encode(inputs)
        
        encoded_features = self.decode(encoded_feature_inputs)
        
        # return self.classifier(encoded_features), encoded_feature_inputs, encoded_features
        return self.classifier(encoded_features)

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        return out


    def decode(self, out):
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        # out = F.relu(self.ident4(out) + out)

        return self.tmp_layer(out)
        
        # return out
    
    def fc_layers(self, out):
        return self.classifier(out)



class SmallResnet50_ft(SmallResnet50):
    
    def __init__(self,in_channels,num_classes):
        super(SmallResnet50_ft, self).__init__(in_channels, num_classes)
        
        # self.stg1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
        #                           nn.BatchNorm2d(64),
        #                           nn.ReLU(inplace=True))
        
    def forward(self,inputs):
        
        encoded_feature_inputs = self.encode(inputs)
        
        encoded_features = self.decode(encoded_feature_inputs)
        
        # return self.classifier(encoded_features), encoded_feature_inputs, encoded_features
        return self.classifier(encoded_features)

    def encode(self, inputs):
        out = self.stg1(inputs)
        
        return out


    def decode(self, out):
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        # out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        # out = F.relu(self.ident3(out) + out)
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        
        out = self.tmp_layer(out)
        
        return out
    
    def fc_layers(self, out):
        return self.classifier(out)


# class ServerResnet(nn.Module):
    
#     def __init__(self,in_channels,num_classes):
#         super().__init__()
        
#         self.stg1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
#                                   nn.BatchNorm2d(64),
#                                   nn.ReLU(inplace=True),
#                                   nn.MaxPool2d(kernel_size=3,stride=2))
        
#         ##stage 2
#         self.convShortcut2 = conv_shortcut(64,256,1)
        
#         self.conv2 = block(64,[64,256],3,1,conv=True)
#         self.ident2 = block(256,[64,256],3,1)

        
#         ##stage 3
#         self.convShortcut3 = conv_shortcut(256,512,2)
        
#         self.conv3 = block(256,[128,512],3,2,conv=True)
#         self.ident3 = block(512,[128,512],3,2)

        
#         ##stage 4
#         self.convShortcut4 = conv_shortcut(512,1024,2)
        
#         self.conv4 = block(512,[256,1024],3,2,conv=True)
#         self.ident4 = block(1024,[256,1024],3,2)
        
#         self.tmp_layer = nn.Sequential(
#             # nn.Sigmoid(),
#             nn.AvgPool2d(kernel_size=(4)),
#             nn.Flatten(),
#             # nn.Linear(1024, args.latent_size)
#         )
        

#         classifier_features = 256


#         if args.one_classifer_flag:
#             self.classifier = nn.Sequential(
#                 nn.Linear(in_features=args.latent_size, out_features=num_classes),
#             )
#         else:
#             ##Classify
#             self.classifier = nn.Sequential(
#                 nn.Linear(in_features=args.latent_size, out_features=classifier_features),
#                 # nn.BatchNorm1d(classifier_features),
#                 # nn.ReLU(),

#                 nn.Linear(in_features=classifier_features,
#                         out_features=classifier_features),

#                 nn.Linear(in_features=classifier_features,
#                         out_features=classifier_features),
        
#                 # nn.BatchNorm1d(classifier_features),
#                 # nn.ReLU(),
#                 nn.Linear(in_features=classifier_features, 
#                         out_features=num_classes)
#             )
        
#     def forward(self,inputs):
        
#         out = self.encode(inputs)
        
#         out = self.decode(out)
        
#         return out

#     def encode(self, inputs):
#         out = self.stg1(inputs)
        
#         #stage 2
#         out = F.relu(self.conv2(out) + self.convShortcut2(out))
#         out = F.relu(self.ident2(out) + out)
#         out = F.relu(self.ident2(out) + out)
        
#         #stage3
#         out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
#         out = F.relu(self.ident3(out) + out)
#         out = F.relu(self.ident3(out) + out)
#         out = F.relu(self.ident3(out) + out)
        
#         #stage4             
#         out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
#         out = F.relu(self.ident4(out) + out)
#         out = F.relu(self.ident4(out) + out)
#         out = F.relu(self.ident4(out) + out)
#         out = F.relu(self.ident4(out) + out)
#         out = F.relu(self.ident4(out) + out)

#         return self.tmp_layer(out)

#     # def reparametrize(self, mu, logvar):
#     #     std = logvar.mul(0.5).exp_()
#     #     if torch.cuda.is_available():
#     #         eps = torch.cuda.FloatTensor(std.size()).normal_()
#     #     else:
#     #         eps = torch.FloatTensor(std.size()).normal_()
#     #     return eps.mul(std).add_(mu)

#     def decode(self, z):
#         # z = torch.cat([z, labels], dim=1)
#         return self.classifier(z)


