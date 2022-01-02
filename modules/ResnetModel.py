
import torch.nn as nn
import torch.nn.functional as F

from parameters import args

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
                                  nn.MaxPool2d(kernel_size=3,stride=2))
        
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
            nn.AvgPool2d(kernel_size=(4)),
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


class SmallResnet50(nn.Module):
    
    def __init__(self,in_channels,num_classes):
        super().__init__()
        
        self.stg1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3,stride=2))
        
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
            nn.AvgPool2d(kernel_size=(4)),
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
        
        # # small stage 4
        # out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
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
        
        # small stage 4
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
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



