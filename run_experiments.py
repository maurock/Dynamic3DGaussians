import yaml
import config
import os

def modify_yaml_config(file_path, new_config):
    """Modify the YAML config file with new settings."""
    # Read the existing YAML file
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update config file with new settings
    for k, v in new_config.items():
        config[k].update(v)
  
    # Write the modified configuration back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(config, file, sort_keys=False)

# Path to YAML file
config_path = os.path.join(os.path.dirname(config.__file__), 'train.yaml')

# Define the new settings
new_config = {
    
}

# Modify the YAML config
modify_yaml_config(config_path, new_config)