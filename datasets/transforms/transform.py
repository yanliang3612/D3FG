TRANSFORM_DICT = {}
def register_transform(name):
    def decorator(cls):
        TRANSFORM_DICT[name] = cls
        return cls
    return decorator