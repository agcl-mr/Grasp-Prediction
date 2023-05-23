import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary 
#from inference.models.grasp_model import GraspModel
from grasp_model import GraspModel
#from torchsummary import summary      #pypi
#from grasp_model import GraspModel

class Encoder(nn.Module):
    def __init__(self, input_channels=4, channel_size=32):
        super().__init__()
        
        self.enc_layer=nn.Sequential(
            nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4),
            nn.ReLU(nn.BatchNorm2d(channel_size)),


            nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(nn.BatchNorm2d(channel_size * 2)),


            nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(nn.BatchNorm2d(channel_size * 4)),

        )
        
    def forward(self, x):
        x = self.enc_layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,output_channel=4,channel_size=32):
        super().__init__()

        self.dec_layer= nn.Sequential(
            nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,output_padding=1),
            nn.ReLU(nn.BatchNorm2d(channel_size * 2)),


            nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,output_padding=1),
            nn.ReLU(nn.BatchNorm2d(channel_size)),


            nn.ConvTranspose2d(channel_size, output_channel, kernel_size=4,padding=2,stride=1),
            nn.ReLU(nn.BatchNorm2d(output_channel)),

        )   
    def forward(self, x):
        x = self.dec_layer(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
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
    
    
class Model(GraspModel):    
    def __init__(self, input_channels=4, output_channels=1,channel_size=32,
                  dropout=False, prob=0.0,imagesize=300):
        super().__init__()
        print('vae model initialised....')
        self.enc=Encoder(input_channels,channel_size)
        self.dec1=Decoder()
        self.dec2=Decoder()
        
        self.imagesize=imagesize
        self.latentDim=int(imagesize/4)
        self.featureDim=int((imagesize/4)*(imagesize/4))

        self.encFC1 = nn.Linear(self.featureDim, self.latentDim)
        self.encFC2 = nn.Linear(self.featureDim, self.latentDim)
        self.decFC1 = nn.Linear(self.latentDim, self.featureDim)

        self.incep1 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)
        self.incep2 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)
        self.incep3 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)
        self.incep4 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)
        self.incep5 = InceptionBlock(128, 64, 32, 32, 16, 16, 16)

        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

        self.pos_output = nn.Conv2d(in_channels=4, out_channels=output_channels, kernel_size=3,padding=1)
        self.cos_output = nn.Conv2d(in_channels=4, out_channels=output_channels, kernel_size=3,padding=1)
        self.sin_output = nn.Conv2d(in_channels=4, out_channels=output_channels, kernel_size=3,padding=1)
        self.width_output = nn.Conv2d(in_channels=4, out_channels=output_channels, kernel_size=3,padding=1)
    
    def reparametrize(self,mu,logvar):         
        std = torch.exp(logvar/2)        # standard deviation
        eps = torch.randn_like(std)
        z=mu + std * eps
        return z       
     
    def forward(self, x_in):    
        x=self.enc(x_in)

        x=x.view(-1,128,self.featureDim)     #changing to 2d array
        mu=self.encFC1(x)                #mean
        logvar=self.encFC2(x)            
        z=self.reparametrize(mu,logvar)
        z = F.relu(self.decFC1(z))
        z = z.view(-1, 128, int(self.imagesize/4), int(self.imagesize/4))     #latent representation

        if self.dropout:
            x_out1=self.dec1(self.dropout1(z))
        else:
            x_out1=self.dec1(z)
        
        
        x = self.incep1(z)
        x = self.incep2(x)
        x = self.incep3(x)
        x = self.incep4(x)
        x = self.incep5(x)

        if self.dropout:
            x_out2=self.dec2(self.dropout2(x))
        else:
            x_out2=self.dec2(x)   

        pos_output = self.pos_output(x_out2)
        cos_output = self.cos_output(x_out2)
        sin_output = self.sin_output(x_out2)
        width_output = self.width_output(x_out2)

        return [pos_output,cos_output,sin_output,width_output,x_out1,mu,logvar]




if __name__ == "__main__":
    #pass
    #enc=Encoder()   #output size=(n,128,56,56)
    #dec=Decoder()   #output size=(n,4,224,224)
    
    Size=(2,4,224,224)
    x=torch.zeros(Size)
    model=Model(imagesize=224)
    #model=model.to('cuda')
    summary(model,Size,device='cpu')
    #model=GenerativeResnet()
    ypred=model(x)
    #print(ypred[0].shape,ypred[1].shape,ypred[2].shape,ypred[3].shape,ypred[4].shape)
    #print(ypred[5].shape)
    loss=model.compute_loss_vae(x,[x,x,x,x,x])
    print(loss)
    #model_stats0=summary(model,(1,4,224,224),verbose=1)
    #model_stats1 = summary(enc,(1,4,224,224),verbose=0)      
    #model_stats2 = summary(dec,(1,128,56,56),verbose=0) 

    #print('s1=',str(model_stats0))    