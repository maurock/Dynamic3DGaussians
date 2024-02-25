import os
import yaml
import config

def modify_yaml_config(new_setting):
    """Modify the YAML config file with new settings."""
    # Read the default YAML file
    with open(os.path.join(os.path.dirname(config.__file__), 'train_default.yaml')) as f:
        config_file = yaml.safe_load(f)

    # Update config file with new settings
    for k, v in new_setting.items():
        config_file[k] = v
  
    return config_file


def save_yaml_config(config_file, file_path):
    """Save the new YAML config file."""
    with open(file_path, 'w') as f:
        yaml.dump(config_file, f)


def modes_to_variables(mode):
    if mode=='img_only':
        transmittance = False
        depth_smoothness = False
    elif mode=='img_smooth':
        transmittance = False
        depth_smoothness = True
    elif mode=='img_smooth_touch':
        transmittance = True
        depth_smoothness = True   
    return transmittance, depth_smoothness 


def experiment_1(settings):
    """Description:
    shiny_img_smooth_touch: 20 touches, 1.0 ratio, 0.1, 0.5 lambda_transmittance, 0.025 lambda_depth_smoothness
    """
    new_settings = []

    # RESULTS
    touches = 20      # <============ CHOOSE THIS 
    ratio = 1.0
    random_selection = False
    modes = ['img_smooth_touch']
    lambda_transmittance = 0.5
    lambda_depth_smoothness = 0.025

    for dataset in ['ShinyBlender', 'GlossySynthetic']:

        if dataset == 'ShinyBlender':
            objs = ['toaster', 'teapot', 'coffee', 'car', 'helmet']
            dataset_abbrv = 'shiny'
        elif dataset == 'GlossySynthetic':
            objs = ['cat', 'angel', 'bell', 'horse', 'luyu', 'potion', 'tbell', 'teapot']
            dataset_abbrv = 'glossy'

        for mode in modes:
            transmittance, depth_smoothness = modes_to_variables(mode)   

            for obj in objs:
                new_settings.append(
                    {
                        # Settings
                        'dataset': f'{dataset}',  # ShinyBlender, GlossySynthetic
                        'input_seq': f'{obj}',  # Path to the directory inside data/, e.g. toaster
                        'output_seq': f'{obj}', # Path to the directory inside output/<exp>, e.g. toaster_15000
                        'exp_name': f'{dataset_abbrv}_{mode}',  # Path to the experiment directory inside output/, e.g. exp1
                        'num_touches': touches,   # Number of touches to simulate
                        'iterations': 30000,  # Number of iterations for the training process
                        'iterations_densify': 15000,  # Number of iterations for the densification process
                        'ratio_data': ratio,  # Ratio of the data to use for training
                        'random_selection': random_selection,

                        # Saving
                        'save_eval_data': False,  # Save evaluation data (True/False)
                        'eval': True,

                        # Mode
                        'transmittance': transmittance,  # Optimise transmittance (True/False)
                        'lambda_transmittance': lambda_transmittance,  # Weight for the transmittance loss
                        'depth_smoothness': depth_smoothness,  # Optimise depth smoothness (True/False)
                        'lambda_depth_smoothness': lambda_depth_smoothness  # Weight for the depth smoothness loss
                    }
                )

    settings.extend(new_settings)

    return settings  



def generate_new_settings():
    """Generate new settings for the experiments. Change this function to generate new settings."""

    new_settings = []

    #################### Add new settings here ##################################

    new_settings = experiment_1(new_settings)

    ###########################################################################

    return new_settings




if __name__=='__main__':

    new_settings = generate_new_settings()
    
    for new_setting in new_settings:
        config_file = modify_yaml_config(new_setting)

        # Save the new YAML config file
        config_path = os.path.join(
            os.path.dirname(config.__file__),
            f'{config_file["exp_name"]}_{config_file["output_seq"]}.yaml'
        )
        save_yaml_config(config_file, config_path)
        