import datetime
import os
import sys
import argparse
import logging
import numpy as np

import cv2
from tqdm import tqdm
import torch
import torch.utils.data
import torch.optim as optim


#from torchsummary import summary
#from torchinfo import summary


from utils.data import get_dataset

from inference.models.test0 import Model0
from inference.models.test1 import Model1
from inference.models.test2 import Model2
from inference.models.test3 import Model3
from inference.models.test4 import Model4
from inference.models.auto_enc import Model

from inference.models.grasp_model import ECA
from inference.models.grconvnet3 import GenerativeResnet

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,default='cornell',
                        help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str,default='D:\DDP\Data\cornell',
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()    
    
    Dataset = get_dataset('cornell')   #CornellDataset(GraspDatasetBase) or jacquarddataset class imported as name =Dataset

    #dataset= object of the class CornellDataset(GraspDatasetBase) or JacquardDataset(GraspDatasetBase)
    path='D:\DDP\Data\jacquard\jacquard_0'
    path='D:\DDP\Data\cornell'

    dataset = Dataset(path,   #dataset= It is object of CornellDataset(GraspDatasetBase) class
                      output_size=224,          #224 for cornell , 300 for jacquard
                      ds_rotate=args.ds_rotate,
                      random_rotate=False,
                      random_zoom=False,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb)
    # dataset[0]= x, (pos, cos, sin, width), idx, rot, zoom_factor = x,y,_,_,_
    
    print(dataset.length)
    

    
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    train_indices, val_indices = indices[:split], indices[split:]
    print('Training size: {}'.format(len(train_indices)))
    print('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)  #gives random sequence of batch for each epoch
    #train_sampler = torch.utils.data.sampler.SequentialSampler(train_indices)    #gives same sequence of batch for each epoch
    

    train_data = torch.utils.data.DataLoader(   #data loader created to load our data into Model in batches,shuffled and using num_workers   
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers
    )    
    #x,y,_,_,_=next(iter(train_data))
    for x,y,_,_,_ in tqdm(train_data):
        pass
    """
    #net=Model()
    for epoch in range(1,4):
        print(f"epoch={epoch}")
        stop=0
        for x,y,_,_,_ in (train_data):
            print(f'Norm:{torch.norm(x)}')
            stop+=1
            if stop==3:
                break
    """


    """
    x_max,x_min=-10**10,10**10
    q_max,q_min=-10**10,10**10
    sin_max,sin_min=-10**10,10**10
    cos_max,cos_min=-10**10,10**10
    w_max,w_min=-10**10,10**10


    q_pred_max,q_pred_min=-10**10,10**10
    sin_pred_max,sin_pred_min=-10**10,10**10
    cos_pred_max,cos_pred_min=-10**10,10**10
    w_pred_max,w_pred_min=-10**10,10**10

    for x,y,_,_,_ in tqdm(train_data):    #here x,y are produced batch wise using data loader
        x_max,x_min=max(x_max,x.max()),min(x_min,x.min())
        q_max,q_min=max(q_max,y[0].max()),min(q_min,y[0].min())
        sin_max,sin_min=max(sin_max,y[1].max()),min(sin_min,y[1].min())
        cos_max,cos_min=max(cos_max,y[2].max()),min(cos_min,y[2].min())
        w_max,w_min=max(w_max,y[3].max()),min(w_min,y[3].min())

        ypred=net(x)
        q_pred_max,q_pred_min=max(q_pred_max,ypred[0].max()),min(q_pred_min,ypred[0].min())
        sin_pred_max,sin_pred_min=max(sin_pred_max,ypred[1].max()),min(sin_pred_min,ypred[1].min())
        cos_pred_max,cos_pred_min=max(cos_pred_max,ypred[2].max()),min(cos_pred_min,ypred[2].min())
        w_pred_max,w_pred_min=max(w_pred_max,ypred[3].max()),min(w_pred_min,ypred[3].min())        

    print('---------------TRAIN DATA --------------')
    print(f"x_max={x_max} , x_min={x_min}")                 #max= 1 , min= -1
    print(f"pos_max={q_max} , pos_min={q_min}")             #max= 1 , min=  0
    print(f"sin_max={sin_max} , sin_min={sin_min}")         #max= 1 , min= -1
    print(f"cos_max={cos_max} , cos_min={cos_min}")         #max= 1 , min= -1
    print(f"width_max={w_max} , width_min={w_min}")         #max= 1 , min=  0        

    print('---------------PRED DATA --------------')
    print(f"pos_max={q_pred_max} , pos_min={q_pred_min}")             #max= 1 , min=  0
    print(f"sin_max={sin_pred_max} , sin_min={sin_pred_min}")         #max= 1 , min= -1
    print(f"cos_max={cos_pred_max} , cos_min={cos_pred_min}")         #max= 1 , min= -1
    print(f"width_max={w_pred_max} , width_min={w_pred_min}")         #max= 1 , min=  0   
    """




    """
    # CHECKING 1 ITERATION
    x,y,_,_,_=next(iter(train_data))
    
    pos=y[0].shape
    sin=y[1].shape
    cos=y[2].shape
    width=y[3].shape
    #x_=y[4].shape


    print('---------------TRAIN DATA --------------')
    print(f"x={x.shape} pos={pos} sin={sin} cos={cos} width={width} ")
    print(f"x_max={x.max()} , x_min={x.min()}")
    print(f"pos_max={y[0].max()} , pos_min={y[0].min()}")        #max=1 , min=0
    print(f"sin_max={y[1].max()} , sin_min={y[1].min()}")         #max=1 , min=-0.99
    print(f"cos_max={y[2].max()} , cos_min={y[2].min()}")        #max=0.97 , min=-0.1
    print(f"width_max={y[3].max()} , width_min={y[3].min()}")       #max=1 , min=0
    #print(f"x_max={y[4].max()} ,x_min={y[4].min()}")               #max=0.24 , min= -0.79---->images are normalised
    

    #net = Model0(input_channels=4, dropout=args.use_dropout, prob=args.dropout_prob,channel_size=args.channel_size)
    ypred=net(x)
    print('---------------PRED DATA --------------')
    print(f"pos_max={ypred[0].max()} , pos_min={ypred[0].min()}")        #max=1 , min=0
    print(f"sin_max={ypred[1].max()} , sin_min={ypred[1].min()}")         #max=1 , min=-0.99
    print(f"cos_max={ypred[2].max()} , cos_min={ypred[2].min()}")        #max=0.97 , min=-0.1
    print(f"width_max={ypred[3].max()} , width_min={ypred[3].min()}")       #max=1 , min=0
    #print(f"x_max={ypred[4].max()} ,x_min={ypred[4].min()}")   
    """

    """
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    
    #net = Model0(input_channels=input_channels, dropout=args.use_dropout, prob=args.dropout_prob,channel_size=args.channel_size)
    net = Model4(input_channels=input_channels, dropout=args.use_dropout, prob=args.dropout_prob,channel_size=args.channel_size)  
    #net=GenerativeResnet(input_channels=input_channels, dropout=args.use_dropout, prob=args.dropout_prob,channel_size=args.channel_size)          

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")                
    #net=net.to(device)   # torchinfo summary works even if model is in cpu, but torchsummary summary needs model in GPU         
    
    #x=torch.ones((5,4,224,224))

    #x=x.to(device) 
    #ypred=net(x)
    #print(ypred[0].shape,ypred[1].shape,ypred[2].shape,ypred[3].shape)

    model_stats = summary(net, (1, 4, 224, 224), verbose=0)
    summary_str = str(model_stats)
    print(summary_str)
    """    







