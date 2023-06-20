#!/usr/bin/python3

import os
import datajoint as dj
import pandas as pd
import numpy as np

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USERNAME']
dj.config['database.password'] = os.environ['DJ_PASSWORD']
dj.config["enable_python_native_blobs"] = True

name = 'vei'
os.environ["DJ_SCHEMA_NAME"] = f"metrics_{name}"
dj.config["nnfabrik.schema_name"] = os.environ["DJ_SCHEMA_NAME"]

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from mei.main import MEISeed, MEIMethod
from nnsysident.tables.mei import MEISelector, TrainedEnsembleModel, MEI
from nnsysident.tables.experiments import *
from nnsysident.tables.scoring import (
    OracleScore,
    R2erScore,
    FeveScore,
    TestCorr,
)
from nnsysident.utility.data_helpers import extract_data_key



# unit_ids = [8,  9, 26, 29, 33, 49, 59, 69, 70, 92]
# ensemble_hash = (TrainedEnsembleModel() & "ensemble_comment = 'gamma models'").fetch1("ensemble_hash")
#
# method_hash = (MEIMethod() & "method_comment like '%MEI%'").fetch1("method_hash")
# MEI().populate(
#     "unit_id in {}".format(tuple(unit_ids)),
#     "method_hash = '{}'".format(method_hash),
#     "ensemble_hash = '{}'".format(ensemble_hash),
#     display_progress=True,
# )
# method_hash = (MEIMethod() & "method_comment like '%CEI%'").fetch1("method_hash")
# MEI().populate(
#     "unit_id in {}".format(tuple(unit_ids)),
#     "method_hash = '{}'".format(method_hash),
#     "ensemble_hash = '{}'".format(ensemble_hash),
#     display_progress=True,
# )
# method_hash = (MEIMethod() & "method_comment like '%VEI%'").fetch("method_hash")
# MEI().populate(
#     "unit_id in {}".format(tuple(unit_ids)),
#     "method_hash in ('{}', '{}')".format(method_hash[0], method_hash[1]),
#     "ensemble_hash = '{}'".format(ensemble_hash),
#     display_progress=True,
# )






### Experiment

# for experiment_name in ['Direct training on transfer dataset']:
#
#     TrainedModel.progress(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModel.populate(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name),
#                           reserve_jobs=True,
#                           order="random",)


### Transfer Experiment

# for experiment_name in ['Transfer between areas (indiv. hyperparams)']:
#
#     TrainedModelTransfer.progress(ExperimentsTransfer.Restrictions & 'seed = 1' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModelTransfer.populate(ExperimentsTransfer.Restrictions & 'seed = 1' & 'experiment_name="{}"'.format(experiment_name),
#                                   reserve_jobs=True,
#                                   order="random",)



######################  Bayesian Search  ###################################
paths = ["/project/notebooks/data/static20457-5-9-preproc0.zip"]
img_data_key = extract_data_key(paths[0])
dataset_fn = "nnsysident.datasets.mouse_loaders.static_loaders"
dataset_config = {'paths': ['/project/notebooks/data/static20457-5-9-preproc0'],
                  'batch_size': 64,
                  'seed': 42,
                  'loader_outputs': ['images', 'responses'],
                  'normalize': True,
                  'exclude': None}
dataset_config_auto = dict()
print(dataset_config)

model_fn = "nnsysident.models.models.stacked2d_zig"
loc = np.exp(-10)
model_config = {
    'zero_thresholds': {img_data_key: loc},
    "init_sigma": 0.4,
    'init_mu_range': 0.55,
    'gamma_input': 1.0,
    'grid_mean_predictor': {'type': 'cortex',
                            'input_dimensions': 2,
                            'hidden_layers': 0,
                            'hidden_features': 0,
                            'final_tanh': False},
    "readout_type": "MultipleGeneralizedFullGaussian2d",
}

print(model_fn)
print(model_config)
model_config_auto = dict(
    feature_reg_weight={"type": "range", "bounds": [1e-2, 1e2], "log_scale": True},
    hidden_channels={"type": "choice", "values": [32, 64, 128, 256]},
    layers={"type": "choice", "values": [3, 4, 5, 6]},
    hidden_kern={"type": "choice", "values": [9, 11, 13, 15, 17]},
    input_kern={"type": "choice", "values": [9, 11, 13, 15, 17]},
)


trainer_fn = "nnsysident.training.trainers.standard_trainer"
trainer_config = {'detach_core': False,
                  'stop_function': 'get_loss',
                  'maximize': False}
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
    model_comment=f"ZIG, {trainer_config['stop_function']}",
)


###########################################################################

# OracleScore.populate(reserve_jobs=True)
# OracleScoreTransfer.populate(reserve_jobs=True)
#
# TestCorr.populate(reserve_jobs=True)
# TestCorrTransfer.populate(reserve_jobs=True)