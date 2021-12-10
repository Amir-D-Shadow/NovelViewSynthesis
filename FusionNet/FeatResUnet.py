import torch
import torch.nn as nn

#Encoder Blocks
class ResUnit(nn.Module):

    def __init__(self,in_channels,mid_channels,out_channels, kernel_size, stride=1,normalization=None,track_running_stats=False):

        """
        Module Graph:

        --------- CNL1 ------ CNL2 ------ Add
            |                              |
            |______________________________|
            
        """
        super(ResUnit,self).__init__()

        self.CNL_1 = CNL(in_channels = in_channels,
                        out_channels = mid_channels,
                        kernel_size = 1,
                        stride = 1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)
        
        self.CNL_2 = CNL(in_channels = mid_channels,
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

    def forward(self,x):

        CNL_1 = self.CNL_1(x)
        CNL_2 = self.CNL_2(CNL_1)

        out = CNL_2 + x

        return out


class CNL(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,normalization=None,track_running_stats=False):
        
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

        self.act1 = nn.LeakyReLU()

    def forward(self,x):

        layer_module = self.conv1(x)

        if self.normalization is not None:
            
            layer_module = self.norm1(layer_module)

        layer_module = self.act1(layer_module)

        return layer_module
        

        
#Fusion Net
class FeatureResUNet(nn.Module):

    def __init__(self,normalization=None,track_running_stats=False):

        super(FeatureResUNet,self).__init__()

        #input 0 in: m x 12 x H x W , out: m x 64 x H//2 x W//2 : connection_0
        self.res0unit = ResUnit(in_channels = 12,
                                mid_channels = 64,
                                out_channels = 64,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL0 = CNL(in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_0 = nn.MaxPool2d(kernel_size=2,stride=2)

        #block 1 in: m x 64 x H//2 x W//2, out: m x 128 x H//4 x W//4 : connection_1
        self.res1unit = ResUnit(in_channels = 64,
                                mid_channels = 128,
                                out_channels = 128,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL1 = CNL(in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_1 = nn.MaxPool2d(kernel_size=2,stride=2)


        #block 2 in: m x 128 x H//4 x W//4, out: m x 256 x H//8 x W//8 : connection_2
        self.res2unit = ResUnit(in_channels = 128,
                                mid_channels = 256,
                                out_channels = 256,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL2 = CNL(in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_2 = nn.MaxPool2d(kernel_size=2,stride=2)

        #block 3 in: m x 256 x H//8 x W//8, out: m x 512 x H//16 x W//16 : connection_3
        self.res3unit = ResUnit(in_channels = 256,
                                mid_channels = 512,
                                out_channels = 512,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL3 = CNL(in_channels=512,
                        out_channels=512,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_3 = nn.MaxPool2d(kernel_size=2,stride=2)

        #BottleNeck in: m x 512 x H//16 x W//16, out: m x 1024 x H//16 x W//16 
        self.BottleNeck_ResUnit = ResUnit(in_channels = 512,
                                          mid_channels = 1024,
                                          out_channels = 1024,
                                          kernel_size = 3,
                                          stride=1,
                                          normalization=normalization,
                                          track_running_stats=track_running_stats)
        
        self.BottleNeck_CNL = CNL(in_channels=1024,
                                  out_channels=1024,
                                  kernel_size=3,
                                  stride=1,
                                  padding="same",
                                  normalization=normalization,
                                  track_running_stats=track_running_stats)
        

        


    def forward(self,x):

        """
        x : (m,C,H,W)
        """

        #input 0 in m x 12 x H x W out: m x 64 x H//2 x W//2 : connection_0
        res0unit = self.res0unit(x)
        CNL0 = self.CNL0(res0unit)
        connection_0 = self.connection_0(CNL0)

        #block 1 in: m x 64 x H//2 x W//2, out: m x 128 x H//4 x W//4 : connection_1
        res1unit = self.res1unit(connection_0)
        CNL1 = self.CNL1(res1unit)
        connection_1 = self.connection_1(CNL1)

        #block 2 in: m x 128 x H//4 x W//4, out: m x 256 x H//8 x W//8 : connection_2
        res2unit = self.res2unit(connection_1)
        CNL2 = self.CNL2(res2unit)
        connection_2 = self.connection_2(CNL2)

        #block 3 in: m x 256 x H//8 x W//8, out: m x 512 x H//16 x W//16 : connection_3
        res3unit = self.res3unit(connection_2)
        CNL3 = self.CNL3(res3unit)
        connection_3 = self.connection_3(CNL3)

        #BottleNeck in: m x 512 x H//16 x W//16, out: m x 1024 x H//16 x W//16 
        BottleNeck_ResUnit = self.BottleNeck_ResUnit(connection_3)
        BottleNeck_CNL = self.BottleNeck_CNL(BottleNeck_ResUnit)



        

        
