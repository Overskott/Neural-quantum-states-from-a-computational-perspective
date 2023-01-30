import json
import sys
import os

# The  config.json file contains the default values used by the code.
module_path = os.path.abspath(os.path.dirname(__file__))
source = 'config.json'

if module_path not in sys.path:
    sys.path.append(f"{module_path}\\{source}")


def get_config_file():
    with open(f"{module_path}\\{source}") as config_file:
        return json.load(config_file)
