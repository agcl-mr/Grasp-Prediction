import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)           # default=F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)         # defualt=F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)         # deafualt=F.smooth_l1_loss(sin_pred, y_sin) 
        width_loss = F.smooth_l1_loss(width_pred, y_width)   # default= F.smooth_l1_loss(width_pred, y_width) 

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
    def compute_loss2(self, xc, yc):              #new loss with reconstructed error loss for auto encoder
        #print('compute_loss2 running......')
        y_pos, y_cos, y_sin, y_width ,x_input= yc

        pos_pred, cos_pred, sin_pred, width_pred ,x_constructed= self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)           # default=F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)         # defualt=F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)         # deafualt=F.smooth_l1_loss(sin_pred, y_sin) 
        width_loss = F.smooth_l1_loss(width_pred, y_width)   # default= F.smooth_l1_loss(width_pred, y_width) 
        reconstructed_loss = F.smooth_l1_loss(x_input, x_constructed)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss+reconstructed_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                'reconstructed_loss':reconstructed_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred,
                'x_constructed':x_constructed
            }
        }
    
    def compute_loss_vae(self, xc, yc):              #new loss with reconstructed error loss for variational auto encoder
        #print('compute_loss2 running......')
        y_pos, y_cos, y_sin, y_width ,x_input= yc

        pos_pred, cos_pred, sin_pred, width_pred ,x_constructed,mu,logvar= self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)           # default=F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)         # defualt=F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)         # deafualt=F.smooth_l1_loss(sin_pred, y_sin) 
        width_loss = F.smooth_l1_loss(width_pred, y_width)   # default= F.smooth_l1_loss(width_pred, y_width) 
        reconstructed_loss = F.smooth_l1_loss(x_input, x_constructed)
        kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss+reconstructed_loss+kl_divergence,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                'reconstructed_loss':reconstructed_loss,
                'kl_divergence':kl_divergence
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred,
                'x_constructed':x_constructed
            }
        }    
    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred ,x_constructed= self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred,
            'x_constructed':x_constructed
        }

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
    

class ECA(nn.Module):
    #Args:channel: Number of channels of the input feature map
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.bn=nn.BatchNorm2d(128)
    def forward(self, x):
        y = self.avg_pool(x)  # feature descriptor on the global spatial information

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)  # Multi-scale information fusion
        return self.bn(x * y.expand_as(x))    