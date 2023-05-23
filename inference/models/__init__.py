def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'model0':
        from .test0 import Model0
        return Model0
    elif network_name == 'model1':
        from .test1 import Model1
        return Model1  
    elif network_name == 'model2':
        from .test2 import Model2
        return Model2        
    elif network_name=='model3':
         from .test3 import Model3
         return Model3    
    elif network_name=='model4':
         from .test4 import Model4
         return Model4
    elif network_name=='model5':
         from .test5 import Model5
         return Model5
    elif network_name == 'auto_enc':
        from .auto_enc import Model
        return Model 
    elif network_name == 'vae':
        from .vae import Model
        return Model    
    elif network_name == 'convnext':
        from .convnext_incep import Model
        return Model           
    elif network_name == 'unet':
        from .unet import unet_model
        return unet_model   
    elif network_name == 'coatnet':
        from .coatnet_incep import coatnet_incep_model
        return coatnet_incep_model       
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
