from neuralpredictors.layers.activations import Elu1
from torch.nn import Identity

from neuralpredictors.layers.encoders.base import Encoder


class GeneralizedEncoderBase(Encoder):
    def __init__(
        self, core, readout, nonlinearity_type_list, shifter=None, modulator=None, nonlinearity_config_list=None
    ):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model. Can predict any distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            nonlinearity_type_list (list of classes/functions): Non-linearity type to use.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are not implemented atm (24/06/2021). Defaults to None.
            nonlinearity_config_list (optional[list of dicts]): Non-linearity configuration. Defaults to None.
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.nonlinearity_type_list = nonlinearity_type_list

        if nonlinearity_config_list is None:
            nonlinearity_config_list = [{}] * len(nonlinearity_type_list)
        self.nonlinearity_config_list = nonlinearity_config_list

    def forward(
        self,
        x,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        # get readout outputs
        x = self.core(x)
        if detach_core:
            x = x.detach()

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"], shift=shift)
        else:
            x = self.readout(x, data_key=data_key, shift=shift)

        if self.modulator:
            x = self.modulator[data_key](x, behavior=behavior)

        assert len(self.nonlinearity_type_list) == len(x) == len(self.nonlinearity_config_list), (
            "Number of non-linearities ({}), number of readout outputs ({}) and, if available, number of non-linearity configs must match. "
            "If you do not wish to restrict a certain readout output with a non-linearity, assign the activation 'Identity' to it."
        )

        output = []
        for i, (nonlinearity, out) in enumerate(zip(self.nonlinearity_type_list, x)):
            output.append(nonlinearity(out, **self.nonlinearity_config_list[i]))

        return tuple(output)


class GaussianEncoder(GeneralizedEncoderBase):
    def __init__(self, core, readout, shifter=None, modulator=None):
        nonlinearity_type_list = [Identity(), Elu1()]
        nonlinearity_config_list = [{}, {"inplace": False}]

        super().__init__(core, readout, nonlinearity_type_list, shifter, modulator, nonlinearity_config_list)

    def predict_mean(self, x, data_key, *args, **kwargs):
        mean, variance = self.forward(x, *args, data_key=data_key, **kwargs)
        return mean

    def predict_variance(self, x, data_key, *args, **kwargs):
        mean, variance = self.forward(x, *args, data_key=data_key, **kwargs)
        return variance


class GammaEncoder(GeneralizedEncoderBase):
    def __init__(self, core, readout, shifter=None, modulator=None):
        nonlinearity_type_list = [Elu1(), Elu1()]
        nonlinearity_config_list = [{"inplace": False}, {"inplace": False}]

        super().__init__(core, readout, nonlinearity_type_list, shifter, modulator, nonlinearity_config_list)

    def predict_mean(self, x, data_key, *args, **kwargs):
        concentration, rate = self.forward(x, *args, data_key=data_key, **kwargs)
        return concentration / rate

    def predict_variance(self, x, data_key, *args, **kwargs):
        concentration, rate = self.forward(x, *args, data_key=data_key, **kwargs)
        return concentration / rate**2
