import torch
#from torchsummary import summary
import torch.nn as nn
from torch import sqrt
import numpy as np
from torchinfo import summary

from inference.models.coatnet_incep import coatnet_incep_model 
from inference.models.coatnet_incep2 import coatnet_incep_model2 
from inference.models.test0 import Model0
from inference.models.test1 import Model1, InceptionBlock
from inference.models.test2 import Model2
from inference.models.test3 import Model3
from inference.models.test4 import Model4
from inference.models.test5 import Model5

from inference.models.auto_enc import Model
from inference.models.grconvnet3 import GenerativeResnet
from inference.models.unet import unet_model
from inference.models.unet2 import unet_model2
from inference.models.coatnet import MBConvBlock,ScaledDotProductAttention

from inference.models.inception_resnetV2 import Inception_ResNetv2


    
if __name__ == "__main__":
    #net=Model5()
    #net=unet_model2()
    #net=MBConvBlock(ksize=3,input_filters=4,output_filters=4,image_size=224)
    #net=ScaledDotProductAttention(channel,channel//8,channel//8)
    #net=InceptionResNetV2Block(in_channels=128)
    #net= Inception_ResNetv2(in_channels=128)
    #net= InceptionBlock(128, 64, 32, 32, 16, 16, 16)
    #net=GenerativeResnet()
    net=coatnet_incep_model(img_size=300) 
    

    #channel=4
    x=torch.ones((2,4,300,300))
    #y=torch.concat([x,x],dim=1)

    #x=x.reshape(1,channel,-1).permute(0,2,1) #B,N,C
    #y=net(x)
    #print(y[0].shape)
    #y=y.reshape(1,channel,int(np.sqrt(y.shape[-2])),int(np.sqrt(y.shape[-2])))

    net=net.to('cuda')
    summary(net, (1,4,224,224))
