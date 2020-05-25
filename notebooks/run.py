import os
import datajoint as dj
import pandas as pd

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['enable_python_native_blobs'] = True

name = "realdata"
dj.config['schema_name'] = f"konstantin_nnsysident_{name}"

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from nnsysident.tables.experiments import *
from nnsysident.tables.scoring import OracleScore



### Experiment

# experiment_name = 'Real, Direct, se2d_fullgaussian2d, 20505-6-1'
#
# TrainedModel.progress(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name))
#
# TrainedModel.populate(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name),
#                       reserve_jobs=True,
#                       order="random",)

## Transfer Experiment

# for experiment_name in ["Real, core_transfer (animal), se2d_fullgaussian2d, 4-set -> 20457-5-9"]:
#
#     TrainedModelTransfer.progress(Seed * ExperimentsTransfer.Restrictions & 'seed=1' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModelTransfer.populate(Seed * ExperimentsTransfer.Restrictions & 'seed=1' & 'experiment_name="{}"'.format(experiment_name),
#                           reserve_jobs=True,
#                           order="random",)



### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

###########      Full Gaussian 2d     ######################

# for neuron_n in [100, 200]: #50, 100, 200, 500, 1000, 3625  ###
#     for image_n in [50]: # 4399, 2500, 1000, 200, 50 ###
#         try:
#             paths = ['data/static22564-2-12-preproc0.zip',
#                      'data/static22564-2-13-preproc0.zip',
#                      'data/static22564-3-8-preproc0.zip',
#                      'data/static22564-3-12-preproc0.zip']
#
#             dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
#             dataset_config = dict(
#                 paths=paths,
#                 batch_size=64,
#                 multi_match_n=neuron_n,
#                 multi_match_base_seed=1,
#                 image_n=image_n,
#                 image_base_seed=1
#
#             )
#             dataset_config_auto = dict()
#             print(dataset_config)
#
#             model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
#             model_config = {"share_features": True,
#                             "init_mu_range": 0.55,
#                             "init_sigma": 0.4,
#                             'input_kern': 15,
#                             'hidden_kern': 13,
#                             'gamma_input': 1.,
#                             'grid_mean_predictor': {'type': 'cortex', 'input_dimensions': 2, 'hidden_layers': 0,
#                                                    'hidden_features': 0, 'final_tanh': False}
#             }
#             print(model_fn)
#             print(model_config)
#             model_config_auto = dict(
#                 gamma_readout={"type": "range", "bounds": [1e-3, 1e3], "log_scale": True},
#             )
#
#
#             trainer_fn = 'nnsysident.training.trainers.standard_trainer'
#             trainer_config = dict()
#             trainer_config_auto = dict(
#                           )
#
#             autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                                  model_fn, model_config, model_config_auto,
#                                  trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                                  trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=20)
#
#             best_parameters, _, _, _ = autobayes.run()
#
#
#             model_config.update(best_parameters['model'])
#             dataset_config.update(best_parameters['dataset'])
#             Model().add_entry(model_fn=model_fn,
#                               model_config=model_config,
#                               model_fabrikant='kklurz',
#                               model_comment='{}, multi_match_n={}, image_n={}'.format(model_fn.split('.')[-1], dataset_config['multi_match_n'], dataset_config['image_n']))
#             Dataset().add_entry(dataset_fn=dataset_fn,
#                                 dataset_config=dataset_config,
#                                 dataset_fabrikant='kklurz',
#                                 dataset_comment='multi_match_n={}, image_n={}'.format(dataset_config['multi_match_n'], dataset_config['image_n']),
#                                 skip_duplicates=True)
#         except:
#             pass


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##########      SpatialxFeatureLinear     ######################
#
# for neuron_n in [500]: # 1000, 3625    ,
#     for image_n in [50]: ###### 50, 200, 500, 1000, 2500
#         print('--------------------------------------')
#         print('Image_n = {}'.format(image_n))
#         print('--------------------------------------')
#     paths = ['data/static22564-2-12-preproc0.zip',
#              'data/static22564-2-13-preproc0.zip',
#              'data/static22564-3-8-preproc0.zip',
#              'data/static22564-3-12-preproc0.zip']
#
#     dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
#     dataset_config = dict(
#         paths=paths,
#         batch_size=64,
#         multi_match_n=neuron_n,
#         multi_match_base_seed=1,
#         image_n=image_n,
#         image_base_seed=1
#
#     )
#     dataset_config_auto = dict()
#     print(dataset_config)
#
#     #   look at model_config -> kernels!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   # "input_kern": 15, "hidden_kern": 15
#
#     model_fn = 'nnsysident.models.models.se2d_spatialxfeaturelinear'
#     model_config = {'gamma_input': 1., "input_kern": 15, "hidden_kern": 13
#     }
#     model_config_auto = dict(
#         gamma_readout={"type": "range", "bounds": [0.001, 0.2], "log_scale": True},
#     )
#
#     trainer_fn = 'nnsysident.training.trainers.standard_trainer'
#     trainer_config = dict()
#     trainer_config_auto = dict(
#                   )
#
#     autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                          model_fn, model_config, model_config_auto,
#                          trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                          trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=30)
#
#     best_parameters, _, _, _ = autobayes.run()
#
#
#     model_config.update(best_parameters['model'])
#     dataset_config.update(best_parameters['dataset'])
#
#     Model().add_entry(model_fn=model_fn,
#                       model_config=model_config,
#                       model_fabrikant='kklurz',
#                       model_comment='{}, multi_match_n={}, image_n={}'.format(model_fn.split('.')[-1], dataset_config['multi_match_n'], dataset_config['image_n']))
#     Dataset().add_entry(dataset_fn=dataset_fn,
#                         dataset_config=dataset_config,
#                         dataset_fabrikant='kklurz',
#                         dataset_comment='multi_match_n={}, image_n={}'.format(dataset_config['multi_match_n'], dataset_config['image_n']),
#                         skip_duplicates=True)



### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


###########      Deterministic Gaussian     ######################
# paths = ['data/static22564-2-12-preproc0.zip',
#          'data/static22564-2-13-preproc0.zip',
#          'data/static22564-3-8-preproc0.zip',
#          'data/static22564-3-12-preproc0.zip']
#
# dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
# dataset_config = dict(
#     paths=paths,
#     batch_size=64,
#     multi_match_n=3625,
#     multi_match_base_seed=1,
#     image_n=4399,
#     image_base_seed=1
#
# )
# dataset_config_auto = dict()
# print(dataset_config)
#
#
# model_fn = 'nnsysident.models.models.se2d_deterministicgaussian2d'
# model_config = {"share_features": True
# }
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-2, 1e2], "log_scale": True},
#     input_kern={"type": "range", "bounds": [8, 18]},
#     hidden_kern={"type": "range", "bounds": [6, 16]},
#     share_features={"type": "choice", "values": [True, False]}
# )
#
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict()
# trainer_config_auto = dict(
#               )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=30)
#
# best_parameters, _, _, _ = autobayes.run()
#
#
# model_config.update(best_parameters['model'])
# dataset_config.update(best_parameters['dataset'])
# Model().add_entry(model_fn=model_fn,
#                   model_config=model_config,
#                   model_fabrikant='kklurz',
#                   model_comment='{}, multi_match_n={}, image_n={}'.format(model_fn.split('.')[-1], dataset_config['multi_match_n'], dataset_config['image_n']))
# Dataset().add_entry(dataset_fn=dataset_fn,
#                     dataset_config=dataset_config,
#                     dataset_fabrikant='kklurz',
#                     dataset_comment='multi_match_n={}, image_n={}'.format(dataset_config['multi_match_n'], dataset_config['image_n']),
#                     skip_duplicates=True)

OracleScore.populate(reserve_jobs=True)
