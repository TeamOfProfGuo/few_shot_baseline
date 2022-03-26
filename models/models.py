import torch


model_dt = {}
def register(name):
    def decorator(cls):
        model_dt[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if name is None:
        return None
    model = model_dt[name](**kwargs)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None):
    if name is None:
        name = 'model'
    model = make(model_sv[name], **model_sv[name + '_args'])
    model.load_state_dict(model_sv[name + '_sd'])
    return model

