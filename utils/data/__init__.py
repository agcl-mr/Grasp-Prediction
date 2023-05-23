#changed cornell_data ---> cornell_data2
#changed jacquard_data ---> jacquard_data2

#use cornell_data to run train_network.py and cornell_data2 for train_network2.py
def get_dataset(dataset_name):     
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
