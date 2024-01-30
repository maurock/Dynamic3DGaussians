import yaml
import config
import os
import train
import torch

def modify_yaml_config(file_path, new_config):
    """Modify the YAML config file with new settings."""
    # Read the existing YAML file
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update config file with new settings
    for k, v in new_config.items():
        config[k] = v
  
    return config


if __name__ == '__main__':
    # Path to YAML file
    config_path = os.path.join(os.path.dirname(config.__file__), 'train.yaml')

    # Define the new settings
    experiments = [
        {
            'input_seq': 'toaster',
            'output_seq': 'toaster_15000',
            'exp_name': 'toaster'
        },
        {
            'input_seq': 'toaster',
            'output_seq': 'toaster_15000_T1',
            'exp_name': 'toaster',
            'transmittance': True,
            'lambda_transmittance': 1
        },
        {
            'input_seq': 'toaster',
            'output_seq': 'toaster_15000_T01',
            'exp_name': 'toaster',
            'transmittance': True,
            'lambda_transmittance': 0.1
        },
        {
            'input_seq': 'toaster',
            'output_seq': 'toaster_15000_T01_smoothness01',
            'exp_name': 'toaster',
            'transmittance': True,
            'lambda_transmittance': 0.1,
            'depth_smoothness': True,
            'lambda_depth_smoothness': 0.1
        },
        {
            'input_seq': 'toaster',
            'output_seq': 'toaster_15000_T1_smoothness01',
            'exp_name': 'toaster',
            'transmittance': True,
            'lambda_transmittance': 1,
            'depth_smoothness': True,
            'lambda_depth_smoothness': 0.1
        },
        {
            'input_seq': 'toaster',
            'output_seq': 'toaster_15000_T1_smoothness1',
            'exp_name': 'toaster',
            'transmittance': True,
            'lambda_transmittance': 1,
            'depth_smoothness': True,
            'lambda_depth_smoothness': 1
        }
    ]


# Modify the YAML config
for exp in experiments:
    new_config = modify_yaml_config(config_path, exp)
    train.main(new_config)
    torch.cuda.empty_cache()