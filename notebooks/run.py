import os
import datajoint as dj
import pandas as pd

dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASS"]
dj.config["enable_python_native_blobs"] = True

name = "interspecies_development"
os.environ["DJ_SCHEMA_NAME"] = f"konstantin_nnsysident_{name}"

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from nnsysident.tables.experiments import *
from nnsysident.tables.scoring import (
    OracleScore,
    OracleScoreTransfer,
    R2erScore,
    R2erScoreTransfer,
    FeveScore,
    FeveScoreTransfer,
)


### Experiment
#
# for experiment_name in ['Direct cores on 1 area each']:
#
#     TrainedModel.progress(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModel.populate(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name),
#                           reserve_jobs=True,
#                           order="random",)


### Transfer Experiment

# for experiment_name in ['Transfer between areas']:
#
#     TrainedModelTransfer.progress(ExperimentsTransfer.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModelTransfer.populate(ExperimentsTransfer.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name),
#                                   reserve_jobs=True,
#                                   order="random",)



#######################  Bayesian Search  ###################################
areas = ["LM"]
paths = ["/notebooks/data/static24391-6-17-GrayImageNet-7bed7f7379d99271be5d144e5e59a8e7.zip"]

dataset_fn = "nnsysident.datasets.mouse_loaders.static_loaders"
dataset_config = {
    "paths": paths,
    "batch_size": 64,
    "seed": 1,
    "file_tree": True,
    "layers": ["L2/3"],
    "areas": areas,
    "neuron_n": 218,
}
dataset_config_auto = dict()
print(dataset_config)

model_fn = "nnsysident.models.models.se2d_fullgaussian2d"
model_config = {
    "pad_input": False,
    "stack": -1,
    # "layers": 4,
    "input_kern": 15,
    "gamma_input": 1,
    # "gamma_readout": 2.439,
    "hidden_dilation": 1,
    "hidden_kern": 13,
    # "hidden_channels": 64,
    "n_se_blocks": 0,
    "depth_separable": True,
    "share_features": False,
    "share_grid": False,
    "init_sigma": 0.4,
    "init_mu_range": 0.55,
    "gauss_type": "full",
    "grid_mean_predictor": {
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 0,
        "hidden_features": 0,
        "final_tanh": False,
    },
}

print(model_fn)
print(model_config)
model_config_auto = dict(
    gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
    hidden_channels={"type": "choice", "values": [64, 128]},
    layers={"type": "choice", "values": [4, 5, 6, 7, 8]},
)


trainer_fn = "nnsysident.training.trainers.standard_trainer"
trainer_config = dict(detach_core=False)
trainer_config_auto = dict()

autobayes = Bayesian(
    dataset_fn,
    dataset_config,
    dataset_config_auto,
    model_fn,
    model_config,
    model_config_auto,
    trainer_fn,
    trainer_config,
    trainer_config_auto,
    architect="kklurz",
    trained_model_table="nnsysident.tables.bayesian.TrainedModelBayesian",
    total_trials=200,
)

best_parameters, _, _, _ = autobayes.run()

model_config.update(best_parameters["model"])
Model().add_entry(
    model_fn=model_fn,
    model_config=model_config,
    model_fabrikant="kklurz",
    model_comment="{} model".format(areas),
)


###########################################################################

OracleScore.populate(reserve_jobs=True)
OracleScoreTransfer.populate(reserve_jobs=True)
