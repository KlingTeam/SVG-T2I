"""Data utilities - use original DataModuleFromConfig"""

from ldm.util import instantiate_from_config


def create_data_module(data_config):
    """Create data module using the original DataModuleFromConfig"""
    return instantiate_from_config(data_config)


def get_dataset_info(data_module):
    """Print dataset information"""
    print("#### Data Information #####")
    if hasattr(data_module, 'datasets'):
        for k in data_module.datasets:
            dataset = data_module.datasets[k]
            print(f"{k}: {dataset.__class__.__name__}, Size: {len(dataset)}")
    else:
        print("Data module structure differs from expected")
    
    return data_module


def setup_data_module(data_module):
    """Setup data module (prepare_data and setup)"""
    data_module.prepare_data()
    data_module.setup()
    return data_module