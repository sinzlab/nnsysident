from collections import OrderedDict
import warnings
import torch
from torch import nn
from torch.nn.init import xavier_normal

class MLP(nn.Module):
    def __init__(self, neurons, input_channels=3, hidden_channels=10, layers=2, **kwargs):
        super().__init__()
        warnings.warn(
            'Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__)
        )
        feat = [nn.Linear(input_channels, hidden_channels), nn.ReLU()]
        for _ in range(layers - 1):
            feat.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU()])
        self.mlp = nn.Sequential(*feat)
        self.linear = nn.Linear(hidden_channels, neurons)

    def regularizer(self):
        return self.linear.weight.abs().mean()

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, x, behavior):
        mod = torch.exp(self.linear(self.mlp(behavior)))
        return x * mod



class StaticModulator(torch.nn.ModuleDict):
    _base_modulator = None

    def __init__(self, n_neurons, input_channels=3, hidden_channels=5,
                 layers=2, gamma_modulator=0, **kwargs):
        warnings.warn(
            'Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__)
        )
        super().__init__()
        self.gamma_modulator = gamma_modulator
        for k, n in n_neurons.items():
            if isinstance(input_channels, OrderedDict):
                ic = input_channels[k]
            else:
                ic = input_channels
            self.add_module(k, self._base_modulator(n, ic, hidden_channels, layers=layers))

    def initialize(self):
        for k, mu in self.items():
            self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_modulator


class MLPModulator(StaticModulator):
    _base_modulator = MLP


def NoModulator(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored
    Returns:
        None
    """
    return None
