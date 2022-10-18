import json

source = 'config.json'

def get_config_file():
    with open('config.json') as config_file:
        return json.load(config_file)
