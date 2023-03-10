import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, model_function, model_config, dataloaders, base_path, seeds, device):
        super().__init__()
        self.model_list = nn.ModuleList()
        for seed in seeds:
            model = model_function(dataloaders, seed, **model_config)
            model.to(device)
            model.load_state_dict(torch.load(base_path + "-seed{}".format(seed)))
            model.eval()
            self.model_list.append(model)

    def predict_mean(self, *args, **kwargs):
        means = []
        for model in self.model_list:
            mean = model.predict_mean(*args, **kwargs)
            means.append(mean)
        means = torch.stack(means)
        return means.mean(0)

    def predict_variance(self, *args, **kwargs):
        means, variances = [], []
        for model in self.model_list:
            means.append(model.predict_mean(*args, **kwargs))
            variances.append(model.predict_variance(*args, **kwargs))
        means = torch.stack(means)
        variances = torch.stack(variances)
        variance = torch.mean(variances + means**2, axis=0) - means.mean(0) ** 2
        return variance
