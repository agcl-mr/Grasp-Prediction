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
    
class InceptionBlock2(nn.Module):
    def __init__(self, in_channels, out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_pool):
        super().__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 2*red_3x3, kernel_size=1, padding=0),
            ConvBlock(2*red_3x3, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, 2*red_5x5, kernel_size=1),
            ConvBlock(2*red_5x5, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1))
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)
    
class Print(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x.shape)
        return x
class InceptionResNetV2Block(nn.Module):
    def __init__(self, in_channels=128, scale=1.0):
        super(InceptionResNetV2Block, self).__init__()

        # Branch 1: 1x1 Convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 Convolution -> 3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution -> 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 56, kernel_size=3, padding=1),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels,192, kernel_size=1),
            nn.BatchNorm2d(192)
        )

        # Scale factor for the residual connection
        self.scale = scale

        self.conv_final=nn.Conv2d(192,128,kernel_size=1)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        # Concatenate the outputs of the branches
        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        concat_output = torch.cat(outputs, 1)

        # Scale the residual connection
        residual_output = self.residual(x) * self.scale

        # Add the residual connection to the concatenated output
        output = concat_output + residual_output
        output = nn.ReLU(inplace=True)(output)
        output=nn.ReLU(inplace=True)(self.conv_final(output))

        return output
    
class unet_model2(GraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.incep1 = InceptionResNetV2Block()
        self.incep2 = InceptionResNetV2Block()
        self.incep3 = InceptionResNetV2Block()
        self.incep4 = InceptionResNetV2Block()
        self.incep5 = InceptionResNetV2Block()

        self.printt=Print()
        #self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        #self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        #self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        #self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        #self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(2*channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=0)  #for unet= multiply by extra 2 for input channels
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(2*channel_size * 2, channel_size, kernel_size=6, stride=2, padding=2,
                                        output_padding=0)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(2*channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x1 = F.relu(self.bn1(self.conv1(x_in)))
        #x = self.printt(x1)
        x2= F.relu(self.bn2(self.conv2(x1)))
        #x = self.printt(x2)
        x3 = F.relu(self.bn3(self.conv3(x2)))
        #x = self.printt(x3)

        x = self.incep1(x3)
        x = self.incep2(x)
        x = self.incep3(x)
        x = self.incep4(x)
        x = self.incep5(x)
        #x = self.printt(x)

        x = torch.cat([x,x3],dim=1)

        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.cat([x,x2],dim=1)
        #x = self.printt(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.cat([x,x1],dim=1)
        #x = self.printt(x)

        x = F.relu(self.conv6(x))
        #x = self.printt(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
