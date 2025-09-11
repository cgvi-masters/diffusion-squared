import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # Double convolution block w/ batch normalization and relu

    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),       # normalizes inputs to have zero mean / unit variance
            nn.ReLU(),                      # inplace=True can be slightly faster but be careful, default is False
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DownLayer(nn.Module):
    # Double conv then max pooling (opposite order from github example)

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = DoubleConv(in_chan, out_chan)
        self.pool = nn.MaxPool2d(2)  #equivalent to nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # return both conv and pooled output
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UpLayer(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2) # upsample
        self.double_conv = DoubleConv(in_chan, out_chan)

    def forward(self, x1, x2):
        x2 = self.up(x2)

        diffY = x1.size(2) - x2.size(2)
        diffX = x1.size(3) - x2.size(3)
        
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x
    

class FiLM(nn.Module):

    def __init__(self, feature_dim, cond_dim):
         super().__init__()
         self.film = nn.Linear(cond_dim, feature_dim*2)   # applies affine transform y = Ax + b

    def forward(self, x, source):
        gamma_beta = self.film(source)              # generates [batch_size, 2*cond_]
        gamma, beta = gamma_beta.chunk(2, dim=1)    # split into two along channel dim
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1] to be broadcasted over feature map [B, C, H, W]
        beta = beta.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1] 

        return gamma * x + beta   # scales features x by gamma and shifts them by beta
    

class UNet(nn.Module):

    def __init__(self, in_chan=1, out_chan=1, cond_dim=1):
        # cond_dim depends on what we pass to film layer

        super().__init__()

        # encoder stack
        self.down1 = DownLayer(in_chan, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256, 512)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        self.film = FiLM(1024, cond_dim)   # apply film layer at bottleneck
        
        # decoder stack
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        # final layer (1x1 conv that maps 64 channels to 1)
        self.final_conv = nn.Conv2d(64, out_chan, kernel_size=1)


    def forward(self, x, cond_vec):
        # input image x: [B, 1, H, W]
        # cond_vec: [B, cond_dim]

        x1, p1 = self.down1(x)   # [B, 64, H/2, W/2]
        x2, p2 = self.down2(p1)  # [B, 128, H/4, W/4]
        x3, p3 = self.down3(p2)  # [B, 256, H/8, H/8]
        x4, p4 = self.down4(p3)  # [B, 512, H/16, H/16]

        b = self.bottleneck(p4)  # [B, 1024, H/16, H/16]
        b = self.film(b, cond_vec) # [B, 1024, H/16, H/16]

        u1 = self.up1(x4, b)     # [B, 512, H/8, H/8]
        u2 = self.up2(x3, u1)    # [B, 256, H/4, W/4]
        u3 = self.up3(x2, u2)    # [B, 128, H/2, W/2]
        u4 = self.up4(x1, u3)    # [B, 64, H, W]

        output = self.final_conv(u4) # [B, 1, H, W]
        
        return output