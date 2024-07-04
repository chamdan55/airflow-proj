import json

def load_dict():
# Opening JSON file
    with open('./data/label_dict.json') as json_file:
        label_dict = json.load(json_file)

    return label_dict