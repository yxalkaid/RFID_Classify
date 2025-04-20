import yaml


def load_yml_config(yml_path):
    with open(yml_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_config(config: dict, pattern):
    group = pattern.split(".")
    node = config
    for key in group:
        if key not in node:
            return None
        node = config[key]
    return node
