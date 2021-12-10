import torch
import torch.nn as nn
import os

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
        
#Decoded Block
class UpCNL(nn.Module):

    def __init__(self,in_channels, out_channels,scale_factor, kernel_size, stride=1,mode="nearest", padding=0,normalization=None,track_running_stats=False):

        super(UpCNL,self).__init__()
        
        self.upsample1 = nn.Upsample(scale_factor=scale_factor,mode=mode)
        
        self.CNL1 = CNL(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        normalization=normalization,
                        track_running_stats=track_running_stats)

    def forward(self,x):

        layer_module = self.upsample1(x)
        
        layer_module = self.CNL1(layer_module)

        return layer_module

    
#Fusion Net
class FeatureResUNet(nn.Module):

    def __init__(self,normalization=None,track_running_stats=False):

        super(FeatureResUNet,self).__init__()

        #input pre-encoded in: m x 12 x H x W , out: m x 32 x H x W
        self.preCNL = CNL(in_channels=12,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding="same",
                          normalization=normalization,
                          track_running_stats=track_running_stats)

        #input 0 in: m x 32 x H x W , out: m x 64 x H//2 x W//2 : connection_0
        self.res0unit = ResUnit(in_channels = 32,
                                mid_channels = 64,
                                out_channels = 32,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL0 = CNL(in_channels=32,
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
                                out_channels = 64,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL1 = CNL(in_channels=64,
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
                                out_channels = 128,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL2 = CNL(in_channels=128,
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
                                out_channels = 256,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL3 = CNL(in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.connection_3 = nn.MaxPool2d(kernel_size=2,stride=2)

        #BottleNeck in: m x 512 x H//16 x W//16, out: m x 512 x H//8 x W//8 
        self.BottleNeck_ResUnit = ResUnit(in_channels = 512,
                                          mid_channels = 1024,
                                          out_channels = 512,
                                          kernel_size = 3,
                                          stride=1,
                                          normalization=normalization,
                                          track_running_stats=track_running_stats)
        
        self.BottleNeck_CNL = CNL(in_channels=512,
                                  out_channels=1024,
                                  kernel_size=3,
                                  stride=1,
                                  padding="same",
                                  normalization=normalization,
                                  track_running_stats=track_running_stats)

        self.BottleNeck_UpCNL = UpCNL(in_channels = 1024,
                                      out_channels = 512,
                                      scale_factor = 2,
                                      kernel_size = 3,
                                      stride = 1,
                                      mode = "nearest",
                                      padding = "same",
                                      normalization=normalization,
                                      track_running_stats=track_running_stats)


        #block 4 (decode block 3) in: m x 1024 x H//8 x W//8 , out: m x 256 x H//4 x W//4
        self.res4unit = ResUnit(in_channels = 1024,
                                mid_channels = 512,
                                out_channels = 1024,
                                kernel_size = 3,
                                stride=1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL4 = CNL(in_channels=1024,
                        out_channels=512,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.UPCNL4 = UpCNL(in_channels = 512,
                            out_channels = 256,
                            scale_factor = 2,
                            kernel_size = 3,
                            stride = 1,
                            mode = "nearest",
                            padding = "same",
                            normalization=normalization,
                            track_running_stats=track_running_stats)

        #block 5 (decode block 2) in: m x 512 x H//4 x W//4 , out: m x 128 x H//2 x W//2
        self.res5unit = ResUnit(in_channels = 512,
                                mid_channels = 256,
                                out_channels = 512,
                                kernel_size = 3,
                                stride = 1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL5 = CNL(in_channels=512,
                        out_channels=256,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.UPCNL5 = UpCNL(in_channels = 256,
                            out_channels = 128,
                            scale_factor = 2,
                            kernel_size = 3,
                            stride = 1,
                            mode = "nearest",
                            padding = "same",
                            normalization=normalization,
                            track_running_stats=track_running_stats)

        #block 6 (decode block 1) in: m x 256 x H//2 x W//2 , out: m x 64 x H x W
        self.res6unit = ResUnit(in_channels = 256,
                                mid_channels = 128,
                                out_channels = 256,
                                kernel_size = 3,
                                stride = 1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL6 = CNL(in_channels=256,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.UPCNL6 = UpCNL(in_channels = 128,
                            out_channels = 64,
                            scale_factor = 2,
                            kernel_size = 3,
                            stride = 1,
                            mode = "nearest",
                            padding = "same",
                            normalization=normalization,
                            track_running_stats=track_running_stats)

        #block 7 (decode input 0) in: m x 128 x H x W , out: m x 3 x H x W
        self.res7unit = ResUnit(in_channels = 128,
                                mid_channels = 64,
                                out_channels = 128,
                                kernel_size = 3,
                                stride = 1,
                                normalization=normalization,
                                track_running_stats=track_running_stats)
        
        self.CNL7 = CNL(in_channels=128,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)

        self.CNL8 = CNL(in_channels=64,
                        out_channels=3,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                        normalization=normalization,
                        track_running_stats=track_running_stats)
        

        
    def forward(self,x):

        """
        x : (m,C,H,W)
        """
        
        #input pre-encoded in: m x 12 x H x W , out: m x 32 x H x W
        preCNL = self.preCNL(x)

        #input 0 in m x 32 x H x W out: m x 64 x H//2 x W//2 : connection_0
        res0unit = self.res0unit(preCNL)
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

        #BottleNeck in: m x 512 x H//16 x W//16, out: m x 512 x H//8 x W//8  
        BottleNeck_ResUnit = self.BottleNeck_ResUnit(connection_3)
        BottleNeck_CNL = self.BottleNeck_CNL(BottleNeck_ResUnit)
        BottleNeck_UpCNL = self.BottleNeck_UpCNL(BottleNeck_CNL)

        # out: m x 1024 x H//8 x W//8 
        cat3 = torch.cat((CNL3,BottleNeck_UpCNL),dim = 1)
        
        #block 4 (decode block 3) in: m x 1024 x H//8 x W//8 , out: m x 256 x H//4 x W//4
        res4unit = self.res4unit(cat3)
        CNL4 = self.CNL4(res4unit)
        UPCNL4 = self.UPCNL4(CNL4)

        # out: m x 512 x H//4 x W//4
        cat2 = torch.cat((CNL2,UPCNL4),dim = 1)

        #block 5 (decode block 2) in: m x 512 x H//4 x W//4 , out: m x 128 x H//2 x W//2
        res5unit = self.res5unit(cat2)
        CNL5 = self.CNL5(res5unit)
        UPCNL5 = self.UPCNL5(CNL5)

        # out: m x 256 x H//2 x W//2
        cat1 = torch.cat((CNL1,UPCNL5),dim = 1)

        #block 6 (decode block 1) in: m x 256 x H//2 x W//2 , out: m x 64 x H x W
        res6unit = self.res6unit(cat1)
        CNL6 = self.CNL6(res6unit)
        UPCNL6 = self.UPCNL6(CNL6)

        # out: m x 128 x H x W
        cat0 = torch.cat((CNL0,UPCNL6),dim = 1)
        
        #block 7 (decode input 0) in: m x 128 x H x W , out: m x 3 x H x W
        res7unit = self.res7unit(cat0)
        CNL7 = self.CNL7(res7unit)

        outCNL = self.CNL8(CNL7)

        return outCNL


if __name__ == "__main__":

    model = FeatureResUNet()
    dummy_input = torch.randn(4, 12, 320, 320)

    torch.onnx.export(model,dummy_input,f"{os.getcwd()}/model_structure.onnx",verbose=True)
    
