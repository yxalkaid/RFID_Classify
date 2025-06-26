import yaml


class RFID_Config:

    def __init__(self, yml_path):
        self.config = load_yml_config(yml_path)

        assert "shape" in self.config
        assert "tags" in self.config
        assert "names" in self.config

    @property
    def shape(self):
        return self.config["shape"]

    @property
    def tags(self):
        return self.config["tags"]

    @property
    def classes(self):
        return self.config["names"]


def load_yml_config(yml_path):
    with open(yml_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_tags(yml_path):
    config = load_yml_config(yml_path)
    return config["tags"]


def get_classes(yml_path):
    config = load_yml_config(yml_path)
    return config["names"]


def get_config(config: dict, pattern):
    group = pattern.split(".")
    node = config
    for key in group:
        if key not in node:
            return None
        node = config[key]
    return node
