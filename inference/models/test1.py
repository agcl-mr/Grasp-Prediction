import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.models.grasp_model import GraspModel, ResidualBlock

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1))
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)    


class Model1(GraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(Model1,self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)
        #self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        #self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)
        #self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.incep1 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)

        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.incep2 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)
        #self.incep2 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.incep3 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)
        #self.incep3 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)

        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.incep4 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)


        self.conv11 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn11 = nn.BatchNorm2d(channel_size * 2)

        self.conv12 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn12 = nn.BatchNorm2d(channel_size)

        self.conv13 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
        
        
        self.conv21 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn21 = nn.BatchNorm2d(channel_size * 2)

        self.conv22 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn22 = nn.BatchNorm2d(channel_size)

        self.conv23 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)


        self.conv31 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn31 = nn.BatchNorm2d(channel_size * 2)

        self.conv32 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn32 = nn.BatchNorm2d(channel_size)

        self.conv33 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
        
        
        self.conv41 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn41 = nn.BatchNorm2d(channel_size * 2)

        self.conv42 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn42 = nn.BatchNorm2d(channel_size)

        self.conv43 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)


        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        #x = self.mp1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        #x = self.mp2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        print('x before res shape------',x.shape)
        """
        x1=self.incep1(self.res1(x))
        x2=self.incep2(self.res2(x))
        x3=self.incep3(self.res3(x))
        x4=self.incep4(self.res4(x))"""

        x1=self.incep1(x)
        x2=self.incep2(x)
        x3=self.incep3(x)
        x4=self.incep4(x)

        x1 = F.relu(self.bn11(self.conv11(x1)))
        x1 = F.relu(self.bn12(self.conv12(x1)))
        x1 = self.conv13(x1)

        x2 = F.relu(self.bn21(self.conv21(x2)))
        x2 = F.relu(self.bn22(self.conv22(x2)))
        x2 = self.conv23(x2)    

        x3 = F.relu(self.bn31(self.conv31(x3)))
        x3 = F.relu(self.bn32(self.conv32(x3)))
        x3 = self.conv33(x3)

        x4 = F.relu(self.bn41(self.conv41(x4)))
        x4 = F.relu(self.bn42(self.conv42(x4)))
        x4 = self.conv43(x4)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x1))
            cos_output = self.cos_output(self.dropout_cos(x2))
            sin_output = self.sin_output(self.dropout_sin(x3))
            width_output = self.width_output(self.dropout_wid(x4))
        else:
            pos_output = self.pos_output(x1)
            cos_output = self.cos_output(x2)
            sin_output = self.sin_output(x3)
            width_output = self.width_output(x4)

        return pos_output, cos_output, sin_output, width_output

