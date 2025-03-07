import torch

def load_checkpoint(model, ckp_path):
    ckp = torch.load(ckp_path, map_location='cpu')
    state_dict = ckp['state_dict']
    new_state_dict = dict()
    for k, v in state_dict.items():
        if 'loss' in k:
            continue
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    return model
