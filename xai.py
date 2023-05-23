import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
from utils.data import get_dataset
#from torchinfo import summary
from torchsummary import summary
from inference.models.test2 import Model2

def transf(x):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)])
    return transform(x)

def func(x,layers):
    outputs = []
    names = []
    temp=None
    for i,layer in enumerate(layers):
            """
            if i==16:
                temp=torch.clone(x)
            elif i>16:
                x=torch.clone(temp)  
            """
            x = layer(x)  
            #if type(layer)==nn.Conv2d:
            outputs.append(x)
            names.append(str(layer))
            print(i,x.shape)
    return outputs,names      

def process(outputs):
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    return processed    

def hook(x,model):
    feature_maps = {}
    inp = x

    def hook_fn(m, i, o):
      feature_maps[m] = o 
      
    net = model

    for name, layer in net._modules.items():
      layer.register_forward_hook(hook_fn)
      
    out = net(inp)

    return feature_maps

def Filter(features):
    output=[]
    names=[]
    for layer,activation in features.items():
        name=str(layer)
        if 'Conv' in name or 'Incep' in name:
            output.append(activation)
            names.append(name[:7])
    return output,names      

def plot(processed,names):
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(6, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        name=names[i]+':'+str(i)
        a.set_title(name, fontsize=30)  
 
    plt.savefig(str('XAI/feature_maps9.jpg'), bbox_inches='tight')        

if __name__=='__main__':
    path="D:/DDP/Projects/2021_Antipodal_robotic-grasping-master/antipodal_logs/221227_1246_training_cornell_model2/epoch_16_iou_0.99"
    #path="D:/DDP/Projects/2021_Antipodal_robotic-grasping-master/antipodal_logs/221228_2011_training_cornell_model3/epoch_27_iou_0.99"
    #model=Model2()
    model = torch.load(path)

    model=model.to('cuda')
    #model_stats = summary(model, (4, 224, 224))   #torchinfo requires 4D size, torchsumary wants 3D size

    Dataset = get_dataset('cornell')   #CornellDataset(GraspDatasetBase) or jacquarddataset class imported as name =Dataset
    path='D:\DDP\Data\cornell'

    train_dataset = Dataset(path,
                      output_size=224,          #224 for cornell , 300 for jacquard
                      ds_rotate=0,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=1,
                      include_rgb=1)
    train_data = torch.utils.data.DataLoader(         
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2)
           
    #path="C:/Users/hrishikesh/Desktop/pcd0115r.png"
    #image = Image.open(path)
    #plt.imshow(image)
    x,y,_,_,_=next(iter(train_data))
    x=x.to('cuda')

    #ypred=model(x)
    #print('length=',len(ypred))
    #x = transf(x)

    #print(f"Image shape before: {x.shape}")
    #$x = x.unsqueeze(0)
    print(f"Image shape: {x.shape}")

    layers=[layer for layer in model.modules()]          #get all layers in common list
    layers2=[layer for layer in model.children()]
    d={i:layer for i,layer in enumerate(layers)}
    d2={i:layer for i,layer in enumerate(layers2)}
    
    #print(layers2)

    feature_maps=hook(x,model)
    output,names=Filter(feature_maps)

    #for output in outputs:
    #    print(output.shape)
    processed=process(output)

    plot(processed,names)

    

