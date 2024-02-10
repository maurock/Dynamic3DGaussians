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
    config_path = os.path.join(os.path.dirname(config.__file__), 'train_default.yaml')

    # ABLATION 1: Performance at increasing number of touches
    # Define the new settings
    experiments = []
    obj = 'toaster'
    touches = [10, 20, 30, 40, 50]      # <============ CHOOSE THIS 
    for touch in touches:
        experiments.append(
            {
                # Settings
                'input_seq': f'{obj}',  # Path to the directory inside data/, e.g. toaster
                'output_seq': f'{obj}_touch{touch}', # Path to the directory inside output/<exp>, e.g. toaster_15000
                'exp_name': f'{obj}_ablation1',  # Path to the experiment directory inside output/, e.g. exp1
                'num_touches': touch,   # Number of touches to simulate
                'iterations': 30000,  # Number of iterations for the training process
                'iterations_densify': 20000,  # Number of iterations for the densification process
                'ratio_data': 1.0,  # Ratio of the data to use for training

                # Saving
                'save_eval_data': True,  # Save evaluation data (True/False)
                'eval': True,

                # Mode
                'transmittance': True,  # Optimise transmittance (True/False)
                'lambda_transmittance': 0.5,  # Weight for the transmittance loss
                'depth_smoothness': True,  # Optimise depth smoothness (True/False)
                'lambda_depth_smoothness': 0.05  # Weight for the depth smoothness loss
            }
        )

    # ABLATION 2: Consistency at equal number of touches
    # Define the new settings
    # experiments = []
    # touches = 30      # <============ CHOOSE THIS 
    # for idx_run in range(0, 8):
    #     experiments.append(
    #         {
    #             # Settings
    #             'input_seq': 'coffee',  # Path to the directory inside data/, e.g. toaster
    #             'output_seq': f'coffee_touch{touches}_{idx_run}', # Path to the directory inside output/<exp>, e.g. toaster_15000
    #             'exp_name': 'coffee_ablation2',  # Path to the experiment directory inside output/, e.g. exp1
    #             'num_touches': touches,   # Number of touches to simulate
    #             'iterations': 30000,  # Number of iterations for the training process
    #             'iterations_densify': 20000,  # Number of iterations for the densification process
    #             'ratio_data': 1.0,  # Ratio of the data to use for training

    #             # Saving
    #             'save_eval_data': True,  # Save evaluation data (True/False)
    #             'eval': True,

    #             # Mode
    #             'transmittance': True,  # Optimise transmittance (True/False)
    #             'lambda_transmittance': 0.5,  # Weight for the transmittance loss
    #             'depth_smoothness': True,  # Optimise depth smoothness (True/False)
    #             'lambda_depth_smoothness': 0.05  # Weight for the depth smoothness loss
    #         }
    #     )


# Modify the YAML config
for exp in experiments:
    new_config = modify_yaml_config(config_path, exp)
    train.main(new_config)
    torch.cuda.empty_cache()