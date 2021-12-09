import torch
import torch.nn as nn

class CNL(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,normalization=None,track_running_stats=None):
        
        """
        normalization : {bn:batch normlaization
                         in:instance normalization
                         ln:layer normalization
                         None: no normalization
                         }
        """
        super(CNL,self).__init__()

        self.normalization = normalization
        self.track_running_stats = track_running_stats
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)

        if self.normalization == "bn":
            
            self.norm1 = nn.BatchNorm2d(out_channels,track_running_stats=self.track_running_stats)

        elif self.normalization == "in":

            self.norm1 = nn.InstanceNorm2d(out_channels,track_running_stats=self.track_running_stats)

        self.act1 = nn.LeakyReLU(inplace=True)

    def forward(self,x):

        layer_module = self.conv1(x)

        if self.normalization is not None:
            
            layer_module = self.norm1(layer_module)

        layer_module = self.act1(layer_module)

        return layer_module
        

        

class FeatureResUNet(nn.Module):

    def __init__(self,normalization=None,track_running_stats=None):

        super(FeatureResUNet,self).__init__()

        #input 0 in: m x 12 x H x W , out: m x 64 x H//2 x W//2 : connection_0
        self.CNL0_1 = CNL(in_channels = 12,
                        out_channels = 64,
                        kernel_size = 1,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)
        
        self.CNL0_2 = CNL(in_channels = 64,
                        out_channels = 64,
                        kernel_size = 3,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_0 = nn.MaxPool2d(kernel_size=2,stride=2)

        #block 1 in: m x 64 x H//2 x W//2 , out: m x 128 x H//4 x W//4 : connection_1
        self.CNL1_1 = CNL(in_channels = 64,
                        out_channels = 128,
                        kernel_size = 1,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)
        
        self.CNL1_2 = CNL(in_channels = 128,
                        out_channels = 128,
                        kernel_size = 3,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_1 = nn.MaxPool2d(kernel_size=2,stride=2)

        #block 2 in: m x 128 x H//4 x W//4 , out: m x 256 x H//8 x W//8 : connection_2
        self.CNL2_1 = CNL(in_channels = 128,
                        out_channels = 256,
                        kernel_size = 1,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)
        
        self.CNL2_2 = CNL(in_channels = 256,
                        out_channels = 256,
                        kernel_size = 3,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_2 = nn.MaxPool2d(kernel_size=2,stride=2)

        #block 3 in: m x 256 x H//8 x W//8 , out: m x 512 x H//16 x W//16 : connection_3
        self.CNL3_1 = CNL(in_channels = 256,
                        out_channels = 512,
                        kernel_size = 1,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)
        
        self.CNL3_2 = CNL(in_channels = 512,
                        out_channels = 512,
                        kernel_size = 3,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_3 = nn.MaxPool2d(kernel_size=2,stride=2)

        

    def forward(self,x):

        """
        x : (m,C,H,W)
        """

        #input 0 in m x 12 x H x W out: m x 64 x H//2 x W//2 : connection_0
        CNL0_1 = self.CNL0_1(x)
        CNL0_2 = self.CNL0_2(CNL0_1)
        connection_0 = self.connection_0(CNL0_2)

        #block 1 in: m x 64 x H//2 x W//2 , out: m x 128 x H//4 x W//4 : connection_1
        CNL1_1 = self.CNL1_1(connection_0)
        CNL1_2 = self.CNL1_2(CNL1_1)
        connection_1 = self.connection_1(CNL1_2)

        #block 2 in: m x 128 x H//4 x W//4 , out: m x 256 x H//8 x W//8 : connection_2
        CNL2_1 = self.CNL2_1(connection_1)
        CNL2_2 = self.CNL2_2(CNL2_1)
        connection_2 = self.connection_2(CNL2_2)

        #block 3 in: m x 256 x H//8 x W//8 , out: m x 512 x H//16 x W//16 : connection_3
        CNL3_1 = self.CNL3_1(connection_2)
        CNL3_2 = self.CNL3_2(CNL3_1)
        connection_3 = self.connection_3(CNL3_2)


        

        
