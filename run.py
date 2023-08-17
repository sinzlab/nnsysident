#!/usr/bin/python3

import os
import time
import datajoint as dj
import pandas as pd
import numpy as np

dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USERNAME"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]
dj.config["enable_python_native_blobs"] = True

name = "vei"
os.environ["DJ_SCHEMA_NAME"] = f"metrics_{name}"
dj.config["nnfabrik.schema_name"] = os.environ["DJ_SCHEMA_NAME"]

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from mei.main import MEISeed, MEIMethod
from nnvision.tables.main import Recording
from nnsysident.tables.mei import (
    MEISelector,
    TrainedEnsembleModel,
    MEI,
    MEIMonkey,
    MEIExperimentsMonkey,
    MEIExperimentsMouse,
)
from nnsysident.tables.experiments import *
from nnsysident.tables.scoring import (
    OracleScore,
    R2erScore,
    FeveScore,
    TestCorr,
)
from nnsysident.utility.data_helpers import extract_data_key

# restr = {'dataset_fn': 'nnsysident.datasets.mouse_loaders.static_loaders',
#          'dataset_hash': '77fecfed4eaa33736d47244f2c14b36b',
#          'model_fn': 'nnsysident.models.models.stacked2d_gamma',
#          'model_hash': 'ea7c8ee30c9f5ab0a632392c3a4b32c0',
#          'trainer_fn': 'nnsysident.training.trainers.standard_trainer',
#          'trainer_hash': '69601593d387758e9ff6a5bf26dd6739'}
# TrainedModel.populate(restr, reserve_jobs=True)
#
# TrainedModel.populate("model_hash = 'c5e4a7ae50f49da6fdff0fb2bce18228'",
#                       "dataset_hash = 'd4869853a4fd946b12adf99b70f9f1cf'",
#                       "trainer_hash = '69601593d387758e9ff6a5bf26dd6739'",
#                       reserve_jobs=True)

MEI.populate(MEIExperimentsMouse.Restrictions & 'experiment_name="{}"'.format("Different L1 weights, CEI (0.8)"),
             reserve_jobs=True,)

########### Mouse MEI
# for experiment_name in ["Zhiwei0, alternative ensemble, OneValue init"]:
#     for mei_type in ["MEI", "CEI", "VEI+", "VEI-"]:
#         restr = (
#                 MEIExperimentsMouse.Restrictions &
#                 (MEIMethod
#                  & f"method_comment like '%{mei_type}%'")
#                 & 'experiment_name="{}"'.format(experiment_name)
#         )
#         MEI().populate(
#             restr,
#             display_progress=True,
#             reserve_jobs=True,
#         )
#         progress = MEI().progress(restr, display=False)
#         while progress[0] != 0:
#             time.sleep(3 * 60)
#             progress = MEI().progress(restr, display=False)


########### Monkey MEI
# ensemble_hash = (TrainedEnsembleModel() &
#                  "ensemble_comment = 'Monkey V1 Gamma Model, PointPooled'").fetch1("ensemble_hash")
# monkey_data_key = "3631807112901"
# monkey_unit_ids = (Recording.Units() & f"data_key = '{monkey_data_key}'").fetch("unit_id")[:10]
#
# for mei_type in ["MEI", "CEI", "VEI+, 0.8", "VEI-, 0.8"]:
#     method_hash = (MEIMethod() & f"method_comment like '%{mei_type}%'").fetch1("method_hash")
#     restr = ["unit_id in {}".format(tuple(monkey_unit_ids)),
#              f"data_key = '{monkey_data_key}'",
#              "method_hash = '{}'".format(method_hash),
#              "ensemble_hash = '{}'".format(ensemble_hash),]
#     MEIMonkey().populate(*restr,
#                          display_progress=True,
#                          reserve_jobs=True,
#                          )
#     progress = MEIMonkey().progress(*restr, display=False)
#     while progress[0] != 0:
#         time.sleep(3*60)
#         progress = MEIMonkey().progress(*restr, display=False)

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


# ######################  Bayesian Search  ###################################
# paths = ["/project/notebooks/data/static20457-5-9-preproc0.zip"]
# img_data_key = extract_data_key(paths[0])
# dataset_fn = "nnsysident.datasets.mouse_loaders.static_loaders"
# dataset_config = {'paths': ['/project/notebooks/data/static20457-5-9-preproc0'],
#                   'batch_size': 64,
#                   'seed': 42,
#                   'loader_outputs': ['images', 'responses'],
#                   'normalize': True,
#                   'exclude': None}
# dataset_config_auto = dict()
# print(dataset_config)
#
# model_fn = "nnsysident.models.models.stacked2d_zig"
# loc = np.exp(-10)
# model_config = {
#     'zero_thresholds': {img_data_key: loc},
#     "init_sigma": 0.4,
#     'init_mu_range': 0.55,
#     'gamma_input': 1.0,
#     'grid_mean_predictor': {'type': 'cortex',
#                             'input_dimensions': 2,
#                             'hidden_layers': 0,
#                             'hidden_features': 0,
#                             'final_tanh': False},
#     "readout_type": "MultipleGeneralizedFullGaussian2d",
# }
#
# print(model_fn)
# print(model_config)
# model_config_auto = dict(
#     feature_reg_weight={"type": "range", "bounds": [1e-2, 1e2], "log_scale": True},
#     hidden_channels={"type": "choice", "values": [32, 64, 128, 256]},
#     layers={"type": "choice", "values": [3, 4, 5, 6]},
#     hidden_kern={"type": "choice", "values": [9, 11, 13, 15, 17]},
#     input_kern={"type": "choice", "values": [9, 11, 13, 15, 17]},
# )
#
#
# trainer_fn = "nnsysident.training.trainers.standard_trainer"
# trainer_config = {'detach_core': False,
#                   'stop_function': 'get_loss',
#                   'maximize': False}
# trainer_config_auto = dict()
#
# autobayes = Bayesian(
#     dataset_fn,
#     dataset_config,
#     dataset_config_auto,
#     model_fn,
#     model_config,
#     model_config_auto,
#     trainer_fn,
#     trainer_config,
#     trainer_config_auto,
#     architect="kklurz",
#     trained_model_table="nnsysident.tables.bayesian.TrainedModelBayesian",
#     total_trials=200,
# )
#
# best_parameters, _, _, _ = autobayes.run()
#
# model_config.update(best_parameters["model"])
# Model().add_entry(
#     model_fn=model_fn,
#     model_config=model_config,
#     model_fabrikant="kklurz",
#     model_comment=f"ZIG, {trainer_config['stop_function']}",
# )


###########################################################################

# OracleScore.populate(reserve_jobs=True)
# OracleScoreTransfer.populate(reserve_jobs=True)
#
# TestCorr.populate(reserve_jobs=True)
# TestCorrTransfer.populate(reserve_jobs=True)


# ################ MONKEY #################################
#
# ######################  Bayesian Search  ###################################
# dataset_fn = "nnvision.datasets.monkey_loaders.monkey_static_loader"
# dataset_config = {'dataset': 'CSRF19_V1',
#                   'neuronal_data_files': [
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3632669014376.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3632932714885.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3633364677437.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3634055946316.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3634142311627.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3634658447291.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3634744023164.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3635178040531.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3635949043110.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3636034866307.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3636552742293.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3637161140869.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3637248451650.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3637333931598.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3637760318484.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3637851724731.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638367026975.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638456653849.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638885582960.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638373332053.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638541006102.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638802601378.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3638973674012.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3639060843972.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3639406161189.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3640011636703.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3639664527524.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3639492658943.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3639749909659.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3640095265572.pickle',
#                       '/project/notebooks/data/monkey/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle'],
#                   'image_cache_path': '/project/notebooks/data/monkey/CSRF19_V1/images/individual',
#                   'crop': 70,
#                   'subsample': 1,
#                   'seed': 1000,
#                   'time_bins_sum': 12,
#                   'batch_size': 128}
# dataset_config_auto = dict()
#
# # model_fn = "nnsysident.models.models.stacked2d_gamma"
# # model_config = {
# #     "init_sigma": 0.4,
# #     'init_mu_range': 0.55,
# #     'gamma_input': 1.0,
# #     'grid_mean_predictor': None,
# #     "readout_type": "MultipleGeneralizedFullGaussian2d",
# # }
# #
# # print(model_fn)
# # print(model_config)
# # model_config_auto = dict(
# #     feature_reg_weight={"type": "range", "bounds": [1e-2, 1e2], "log_scale": True},
# #     hidden_channels={"type": "choice", "values": [32, 64, 128, 256]},
# #     layers={"type": "choice", "values": [3, 4, 5, 6]},
# #     hidden_kern={"type": "choice", "values": [9, 11, 13, 15, 17]},
# #     input_kern={"type": "choice", "values": [9, 11, 13, 15, 17]},
# #     gamma_input={"type": "range", "bounds": [0.1, 10.]}
# # )
#
#
# model_fn = "nnsysident.models.models.stacked2d_gamma"
# model_config = {
#     "readout_type": "MultipleGeneralizedPointPooled2d",
#     'hidden_dilation': 2,
# }
#
# print(model_fn)
# print(model_config)
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-2, 1e2], "log_scale": True},
#     hidden_channels={"type": "choice", "values": [20, 32, 64, 128]},
#     layers={"type": "choice", "values": [3, 4, 5, 6]},
#     hidden_kern={"type": "choice", "values": [7, 9, 11, 13, 15]},
#     input_kern={"type": "choice", "values": [9, 15, 20, 24, 30]},
#     gamma_input={"type": "range", "bounds": [0.1, 100.], "log_scale": True}
# )
#
#
# trainer_fn = "nnsysident.training.trainers.standard_trainer"
# trainer_config = {'detach_core': False,
#                   'stop_function': 'get_correlations',
#                   'maximize': True}
# trainer_config_auto = dict()
#
# autobayes = Bayesian(
#     dataset_fn,
#     dataset_config,
#     dataset_config_auto,
#     model_fn,
#     model_config,
#     model_config_auto,
#     trainer_fn,
#     trainer_config,
#     trainer_config_auto,
#     architect="kklurz",
#     trained_model_table="nnsysident.tables.bayesian.TrainedModelBayesian",
#     total_trials=200,
# )
#
# best_parameters, _, _, _ = autobayes.run()
#
# model_config.update(best_parameters["model"])
# Model().add_entry(
#     model_fn=model_fn,
#     model_config=model_config,
#     model_fabrikant="kklurz",
#     model_comment=f"Gamma, monkey, {trainer_config['stop_function']}",
# )
