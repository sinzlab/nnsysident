import copy

import numpy as np

from neuralpredictors.layers.cores import Stacked2dCore
from neuralpredictors.layers.encoders import FiringRateEncoder, GammaEncoder, GaussianEncoder, ZIGEncoder, ZILEncoder
from neuralpredictors.layers.modulators.mlp import MLPModulator
from neuralpredictors.layers.readouts import (
    FullFactorized2d,
    FullGaussian2d,
    GeneralizedFullGaussianReadout2d,
    GeneralizedPointPooled2d,
    MultiReadoutBase,
    MultiReadoutSharedParametersBase,
    PointPooled2d,
)
from neuralpredictors.layers.shifters import MLPShifter
from neuralpredictors.utils import get_module_output

from ..utility.data_helpers import get_dims_for_loader_dict, get_mean_activity_dict, set_random_seed, unpack_data_info


class MultiplePointPooled2d(MultiReadoutBase):
    _base_readout = PointPooled2d


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d


class MultipleSpatialXFeatureLinear(MultiReadoutBase):
    _base_readout = FullFactorized2d


class MultipleFullSXF(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d


class MultipleFullFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d


class MultipleGeneralizedFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = GeneralizedFullGaussianReadout2d


class MultipleGeneralizedPointPooled2d(MultiReadoutBase):
    _base_readout = GeneralizedPointPooled2d


class Stacked2dCoreReadoutModel:
    def __init__(self):
        self.readout_type = None

    def build_base_model(
        self,
        dataloaders,
        seed,
        data_info=None,
        transfer_state_dict=None,
        # core args
        hidden_channels=64,
        input_kern=9,
        hidden_kern=7,
        layers=4,
        gamma_input=None,
        skip=0,
        bias=False,
        final_nonlinearity=True,
        momentum=0.9,
        pad_input=False,
        batch_norm=True,
        batch_norm_scale=True,
        independent_bn_bias=False,
        hidden_dilation=1,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        stack=-1,
        depth_separable=True,
        linear=False,
        # encoder args
        modulator_kwargs=None,
        shifter_kwargs=None,
        # general readout args
        readout_bias=True,
        gamma_readout=None,
        feature_reg_weight=0.0076,
        inferred_params_n=1,
        # gaussian readout
        init_mu_range=0.3,
        init_sigma=0.1,
        gauss_type="full",
        grid_mean_predictor={
            "type": "cortex",
            "input_dimensions": 2,
            "hidden_layers": 0,
            "hidden_features": 30,
            "final_tanh": True,
        },
        share_features=False,
        share_grid=False,
        share_transform=False,
        init_noise=1e-3,
        init_transform_scale=0.2,
        # pointpooled readout
        pool_steps=2,
        pool_kern=3,
        init_range=0.2,
        **kwargs,
    ):
        if gamma_readout is not None:
            feature_reg_weight = gamma_input

        if transfer_state_dict is not None:
            print(
                "Transfer state_dict given. This will only have an effect in the bayesian hypersearch. See: TrainedModelBayesianTransfer "
            )

        if data_info is not None:
            n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
            mean_activity_dict = None
        else:
            if "train" in dataloaders.keys():
                dataloaders = dataloaders["train"]

            # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
            in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

            mean_activity_dict = get_mean_activity_dict(dataloaders) if readout_bias else None
            session_shape_dict = get_dims_for_loader_dict(dataloaders)
            n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
            in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
            input_channels = [v[in_name][1] for v in session_shape_dict.values()]

        core_input_channels = (
            list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]
        )

        set_random_seed(seed)

        core = Stacked2dCore(
            input_channels=core_input_channels,
            hidden_channels=hidden_channels,
            input_kern=input_kern,
            hidden_kern=hidden_kern,
            layers=layers,
            gamma_input=gamma_input,
            skip=skip,
            final_nonlinearity=final_nonlinearity,
            bias=bias,
            momentum=momentum,
            pad_input=pad_input,
            batch_norm=batch_norm,
            hidden_dilation=hidden_dilation,
            laplace_padding=laplace_padding,
            input_regularizer=input_regularizer,
            stack=stack,
            depth_separable=depth_separable,
            linear=linear,
            batch_norm_scale=batch_norm_scale,
            independent_bn_bias=independent_bn_bias,
        )
        in_shape_dict = {k: get_module_output(core, in_shape)[1:] for k, in_shape in in_shapes_dict.items()}

        if self.readout_type == "MultipleGeneralizedFullGaussian2d":
            source_grids = None
            grid_mean_predictor_type = None
            if grid_mean_predictor is not None:
                grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
                grid_mean_predictor_type = grid_mean_predictor.pop("type")
                if grid_mean_predictor_type == "cortex":
                    input_dim = grid_mean_predictor.pop("input_dimensions", 2)
                    source_grids = {}
                    for k, v in dataloaders.items():
                        # real data
                        try:
                            if v.dataset.neurons.animal_ids[0] != 0:
                                source_grids[k] = v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                            # simulated data -> get random linear non-degenerate transform of true positions
                            else:
                                source_grid_true = v.dataset.neurons.center[:, :input_dim]
                                det = 0.0
                                loops = 0
                                grid_bias = np.random.rand(2) * 3
                                while det < 5.0 and loops < 100:
                                    matrix = np.random.rand(2, 2) * 3
                                    det = np.linalg.det(matrix)
                                    loops += 1
                                assert det > 5.0, "Did not find a non-degenerate matrix"
                                source_grids[k] = np.add((matrix @ source_grid_true.T).T, grid_bias)
                        except FileNotFoundError:
                            print("Dataset type is not recognized to be from Baylor College of Medicine.")
                            source_grids[k] = v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                elif grid_mean_predictor_type == "shared":
                    pass
                else:
                    raise ValueError("Grid mean predictor type {} not understood.".format(grid_mean_predictor_type))

            shared_match_ids = None
            if share_features or share_grid:
                shared_match_ids = {k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()}
                all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

                for match_id in shared_match_ids.values():
                    assert len(set(match_id) & all_multi_unit_ids) == len(
                        all_multi_unit_ids
                    ), "All multi unit IDs must be present in all datasets"

            readout = MultipleGeneralizedFullGaussian2d(
                in_shape_dict=in_shape_dict,
                n_neurons_dict=n_neurons_dict,
                mean_activity_dict=mean_activity_dict,
                init_mu_range=init_mu_range,
                bias=readout_bias,
                init_sigma=init_sigma,
                feature_reg_weight=feature_reg_weight,
                gauss_type=gauss_type,
                grid_mean_predictor=grid_mean_predictor,
                grid_mean_predictor_type=grid_mean_predictor_type,
                source_grids=source_grids,
                share_features=share_features,
                share_grid=share_grid,
                share_transform=share_transform,
                shared_match_ids=shared_match_ids,
                init_noise=init_noise,
                init_transform_scale=init_transform_scale,
                inferred_params_n=inferred_params_n,
            )

        elif self.readout_type == "MultipleGeneralizedPointPooled2d":
            readout = MultipleGeneralizedPointPooled2d(
                in_shape_dict=in_shape_dict,
                n_neurons_dict=n_neurons_dict,
                mean_activity_dict=mean_activity_dict,
                pool_steps=pool_steps,
                pool_kern=pool_kern,
                bias=readout_bias,
                gamma_readout=gamma_readout,
                init_range=init_range,
                inferred_params_n=inferred_params_n,
            )
        else:
            raise ValueError("Readout Type not defined")

        if modulator_kwargs is None:
            modulator = None
        else:
            modulator = MLPModulator(n_neurons_dict, **modulator_kwargs, n_parameters_to_modulate=inferred_params_n)

        if shifter_kwargs is None:
            shifter = None
        else:
            shifter = MLPShifter(list(n_neurons_dict.keys()), **shifter_kwargs)

        return core, readout, shifter, modulator


class Stacked2dPointPooled_Poisson(Stacked2dCoreReadoutModel):
    def __init__(self):
        super().__init__()
        self.readout_type = "MultipleGeneralizedPointPooled2d"

    def build_model(self, dataloaders, seed, elu_offset=0, **kwargs):
        inferred_params_n = 1
        core, readout, shifter, modulator = self.build_base_model(
            dataloaders, seed, inferred_params_n=inferred_params_n, **kwargs
        )

        model = FiringRateEncoder(
            core=core, readout=readout, shifter=shifter, modulator=modulator, elu_offset=elu_offset
        )

        return model


class Stacked2dPointPooled_Gamma(Stacked2dCoreReadoutModel):
    def __init__(self):
        super().__init__()
        self.readout_type = "MultipleGeneralizedPointPooled2d"

    def build_model(self, dataloaders, seed, eps=1.e-6, **kwargs):
        inferred_params_n = 2
        core, readout, shifter, modulator = self.build_base_model(
            dataloaders, seed, inferred_params_n=inferred_params_n, **kwargs
        )

        model = GammaEncoder(core=core, readout=readout, shifter=shifter, modulator=modulator, eps=eps)

        return model


class Stacked2dPointPooled_Gaussian(Stacked2dCoreReadoutModel):
    def __init__(self):
        super().__init__()
        self.readout_type = "MultipleGeneralizedPointPooled2d"

    def build_model(self, dataloaders, seed, eps=1.e-6, **kwargs):
        inferred_params_n = 2
        core, readout, shifter, modulator = self.build_base_model(
            dataloaders, seed, inferred_params_n=inferred_params_n, **kwargs
        )

        model = GaussianEncoder(core=core, readout=readout, shifter=shifter, modulator=modulator, eps=eps)

        return model


class Stacked2dFullGaussian2d_Poisson(Stacked2dCoreReadoutModel):
    def __init__(self):
        super().__init__()
        self.readout_type = "MultipleGeneralizedFullGaussian2d"

    def build_model(self, dataloaders, seed, elu_offset=0, **kwargs):
        inferred_params_n = 1
        core, readout, shifter, modulator = self.build_base_model(
            dataloaders, seed, inferred_params_n=inferred_params_n, **kwargs
        )

        model = FiringRateEncoder(
            core=core, readout=readout, shifter=shifter, modulator=modulator, elu_offset=elu_offset
        )

        return model


class Stacked2dFullGaussian2d_ZIG(Stacked2dCoreReadoutModel):
    def __init__(self):
        super().__init__()
        self.readout_type = "MultipleGeneralizedFullGaussian2d"

    def build_model(
        self,
        dataloaders,
        seed,
        zero_thresholds=None,
        init_ks=None,
        theta_image_dependent=True,
        k_image_dependent=True,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
        **kwargs,
    ):
        inferred_params_n = theta_image_dependent + k_image_dependent + loc_image_dependent + q_image_dependent

        if zero_thresholds == "from dataset":
            zero_thresholds = {}
            for k, v in dataloaders["train"].items():
                zero_thresholds[k] = v.dataset.neurons.normalized_zero_thresholds

        if init_ks == "from dataset":
            init_ks = {}
            for k, v in dataloaders["train"].items():
                init_ks[k] = v.dataset.neurons.normalized_ks
        core, readout, shifter, modulator = self.build_base_model(
            dataloaders, seed, inferred_params_n=inferred_params_n, **kwargs
        )

        model = ZIGEncoder(
            core,
            readout,
            zero_thresholds=zero_thresholds,
            init_ks=init_ks,
            theta_image_dependent=theta_image_dependent,
            k_image_dependent=k_image_dependent,
            loc_image_dependent=loc_image_dependent,
            q_image_dependent=q_image_dependent,
            shifter=shifter,
            modulator=modulator,
            offset=offset,
        )
        return model


class Stacked2dFullGaussian2d_ZIL(Stacked2dCoreReadoutModel):
    def __init__(self):
        super().__init__()
        self.readout_type = "MultipleGeneralizedFullGaussian2d"

    def build_model(
        self,
        dataloaders,
        seed,
        zero_thresholds=None,
        mu_image_dependent=True,
        sigma2_image_dependent=True,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
        **kwargs,
    ):
        inferred_params_n = mu_image_dependent + sigma2_image_dependent + loc_image_dependent + q_image_dependent

        if zero_thresholds == "from dataset":
            zero_thresholds = {}
            for k, v in dataloaders["train"].items():
                zero_thresholds[k] = v.dataset.neurons.normalized_zero_thresholds

        core, readout, shifter, modulator = self.build_base_model(
            dataloaders, seed, inferred_params_n=inferred_params_n, **kwargs
        )

        model = ZILEncoder(
            core,
            readout,
            zero_thresholds=zero_thresholds,
            mu_image_dependent=mu_image_dependent,
            sigma2_image_dependent=sigma2_image_dependent,
            loc_image_dependent=loc_image_dependent,
            q_image_dependent=q_image_dependent,
            shifter=shifter,
            modulator=modulator,
            offset=offset,
        )
        return model
