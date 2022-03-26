import os


DEFAULT_ROOT = './materials'


datasets_dt = {}
def register(name):
    def decorator(cls):
        datasets_dt[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name)
    dataset = datasets_dt[name](**kwargs)
    return dataset

