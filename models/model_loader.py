_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls
    return decorator

def get_model(config, device):
    return _MODEL_DICT[config.type](config).to(device)