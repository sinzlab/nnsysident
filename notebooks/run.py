import os
import datajoint as dj
import pandas as pd

dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASS"]
dj.config["enable_python_native_blobs"] = True

name = "iclr"  # "test" #"realdata"
dj.config["schema_name"] = f"konstantin_nnsysident_{name}"

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from nnsysident.tables.experiments import *
from nnsysident.tables.scoring import OracleScore, OracleScoreTransfer, R2erScore, R2erScoreTransfer, FeveScore, FeveScoreTransfer



### Experiment

for experiment_name in ['Real, Direct, se2d_fullgaussian2d, 20457-5-9, AnscombeLoss']:

    TrainedModel.progress(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name))

    TrainedModel.populate(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name),
                          reserve_jobs=True,
                          order="random",)
#
#
# # Transfer Experiment
#
# for experiment_name in list(ExperimentsTransfer().fetch('experiment_name')):
#
#     TrainedModelTransfer.progress(Seed * ExperimentsTransfer.Restrictions & 'seed=1' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModelTransfer.populate(Seed * ExperimentsTransfer.Restrictions & 'seed=1' & 'experiment_name="{}"'.format(experiment_name),
#                           reserve_jobs=True,
#                           order="random",)


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# #########      Full Gaussian 2d     ######################
#
# for neuron_n in [1000]: #   #50, 500, 3625
#     for image_n in [4399]: # [ 50, 200, 500, 1000, 2500, 4399]
#         # try:
#         paths = ['data/static22564-2-12-preproc0.zip',
#                  'data/static22564-2-13-preproc0.zip',
#                  'data/static22564-3-8-preproc0.zip',
#                  'data/static22564-3-12-preproc0.zip']
#
#         dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
#         dataset_config = dict(
#             paths=paths,
#             batch_size=64,
#             multi_match_n=neuron_n,
#             multi_match_base_seed=1,
#             image_n=image_n,
#             image_base_seed=1
#
#         )
#         dataset_config_auto = dict()
#         print(dataset_config)
#
#         model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
#         model_config = {#"share_features": True,
#                         #"share_transform":True,
#                         "init_mu_range": 0.55,
#                         "init_sigma": 0.4,
#                         'input_kern': 15,
#                         'hidden_kern': 13,
#                         'gamma_input': 1.,
#                         'grid_mean_predictor': {'type': 'cortex', 'input_dimensions': 2, 'hidden_layers': 0, 'hidden_features': 0, 'final_tanh': False},
#         }
#         print(model_fn)
#         print(model_config)
#         model_config_auto = dict(
#             gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
#             share_features={"type": "choice", "values": [True, False]},
#             share_transform={"type": "choice", "values": [True, False]},
#             # init_transform_scale={"type": "range", "bounds": [0.1, 0.7]},
#         )
#
#
#         trainer_fn = 'nnsysident.training.trainers.standard_trainer'
#         trainer_config = dict(detach_core=False)
#         trainer_config_auto = dict(
#                       )
#
#         autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                              model_fn, model_config, model_config_auto,
#                              trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                              trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
#         best_parameters, _, _, _ = autobayes.run()
#
#         except:
#             pass
#         model_config.update(best_parameters['model'])
#         dataset_config.update(best_parameters['dataset'])
#
#         Model().add_entry(model_fn=model_fn,
#                           model_config=model_config,
#                           model_fabrikant='kklurz',
#                           model_comment='{}, multi_match_n={}, image_n={}'.format(model_fn.split('.')[-1], dataset_config['multi_match_n'], dataset_config['image_n']))
#         Dataset().add_entry(dataset_fn=dataset_fn,
#                             dataset_config=dataset_config,
#                             dataset_fabrikant='kklurz',
#                             dataset_comment='multi_match_n={}, image_n={}'.format(dataset_config['multi_match_n'], dataset_config['image_n']),
#                             skip_duplicates=True)


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# #########      SpatialxFeatureLinear     ######################
#
# for neuron_n in [50]: # 50, 500, 1000, 3625    ,
#     for image_n in [50]: ###### 50, 200, 500, 1000, 2500, 4399
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
#     model_config = {'gamma_input': 1., "input_kern": 15, "hidden_kern": 13, "normalize":True
#     }
#     model_config_auto = dict(
#         gamma_readout={"type": "range", "bounds": [0.001, 20.], "log_scale": True},
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
#                          trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
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


##########      Task driven     ######################

# paths = ['data/static20457-5-9-preproc0.zip']
# # [50, 200, 500, 1000, 2500, 4472]
# dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
# dataset_config = dict(
#     paths=paths,
#     batch_size=64,
#     neuron_n=5335,
#     neuron_base_seed=1,
#     image_n=4472,
#     image_base_seed=1,
#     exclude=[],
# )
# dataset_config_auto = dict()
# print(dataset_config)
#
#
# model_fn = 'nnsysident.models.models.taskdriven_fullSXF'
# model_config = {'tl_model_name': "vgg16",
#                 'layers': 8,
#                  'pretrained': True,
#                  'normalize': False,
#                  'share_features': False,
#                 }
#
#
#
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [0.0001, 0.5], "log_scale": True}
# )
#
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict(detach_core=True)
# trainer_config_auto = dict(
#               )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
# best_parameters, _, _, _ = autobayes.run()
#
#
# model_config.update(best_parameters['model'])
# dataset_config.update(best_parameters['dataset'])
# Model().add_entry(model_fn=model_fn,
#                   model_config=model_config,
#                   model_fabrikant='kklurz',
#                   model_comment='{}, neuron_n={}, image_n={}, pretrained={}'.format(model_fn.split('.')[-1], dataset_config['neuron_n'], dataset_config['image_n'], model_config['pretrained']))
# Dataset().add_entry(dataset_fn=dataset_fn,
#                     dataset_config=dataset_config,
#                     dataset_fabrikant='kklurz',
#                     dataset_comment='neuron_n={}, image_n={}, exclude={}'.format(dataset_config['neuron_n'], dataset_config['image_n'], dataset_config['exclude']),
#                     skip_duplicates=True)


#### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

########      SXF     ######################

# for neuron_n in [5335]: # 50, 500, 1000, 3625    ,
#     for image_n in [4472]: ###### 50, 200, 500, 1000, 2500, 4472
#         print('--------------------------------------')
#         print('Image_n = {}'.format(image_n))
#         print('--------------------------------------')
#         paths = ['data/static20457-5-9-preproc0.zip']
#
#         dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
#         dataset_config = dict(
#             paths=paths,
#             batch_size=64,
#             neuron_n=neuron_n,
#             neuron_base_seed=1,
#             image_n=image_n,
#             image_base_seed=1
#
#         )
#         dataset_config_auto = dict()
#         print(dataset_config)
#
#
#         model_fn = 'nnsysident.models.models.se2d_fullSXF'
#         model_config = {'gamma_input': 1.,
#                         "input_kern": 15,
#                         "hidden_kern": 13,
#                         'normalize': False,
#                         'share_features': False,
#         }
#         model_config_auto = dict(
#             gamma_readout={"type": "range", "bounds": [0.0001, 0.5], "log_scale": True},
#             #share_features={"type": "choice", "values": [True, False]}
#         )
#
#         trainer_fn = 'nnsysident.training.trainers.standard_trainer'
#         trainer_config = dict()
#         trainer_config_auto = dict(
#                       )
#
#         autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                              model_fn, model_config, model_config_auto,
#                              trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                              trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
#         best_parameters, _, _, _ = autobayes.run()
#
#
#         model_config.update(best_parameters['model'])
#         dataset_config.update(best_parameters['dataset'])
#
#         Model().add_entry(model_fn=model_fn,
#                           model_config=model_config,
#                           model_fabrikant='kklurz',
#                           model_comment='{}, neuron_n={}, image_n={}'.format(model_fn.split('.')[-1], dataset_config['neuron_n'], dataset_config['image_n']))
#         Dataset().add_entry(dataset_fn=dataset_fn,
#                             dataset_config=dataset_config,
#                             dataset_fabrikant='kklurz',
#                             dataset_comment='neuron_n={}, image_n={}'.format(dataset_config['neuron_n'], dataset_config['image_n']),
#                             skip_duplicates=True)


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

###########      Full Gaussian 2d TRANSFER    ######################

# for neuron_n in [1000]:  # 1000,
#     for image_n in [1000]:  # [ 50, 200, 500, 1000, 2500, 4399]
#         # try:
#         paths = ['data/static22564-2-12-preproc0.zip',
#                  'data/static22564-2-13-preproc0.zip',
#                  'data/static22564-3-8-preproc0.zip',
#                  'data/static22564-3-12-preproc0.zip']
#
# # for neuron_n in [5335]:  # 5335,
# #     for image_n in [1000]:  # [ 50, 200, 500, 1000, 2500, 4472]
# #         paths = ['data/static20457-5-9-preproc0.zip']
#
#
#         dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
#
#         dataset_config = dict(
#             paths=paths,
#             batch_size=64,
#             multi_match_n=neuron_n,
#             multi_match_base_seed=1,
#             image_n=image_n,
#             image_base_seed=1
#         )
#
#         # dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
#         #
#         # dataset_config = dict(
#         #     paths=paths,
#         #     batch_size=64,
#         #     neuron_n=neuron_n,
#         #     neuron_base_seed=1,
#         #     image_n=image_n,
#         #     image_base_seed=1
#         # )
#
#
#         dataset_config_auto = dict()
#         print(dataset_config)
#
#         model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
#         model_config = {
#             "init_mu_range": 0.55,
#             "init_sigma": 0.4,
#             'input_kern': 15,
#             'hidden_kern': 13,
#             'gamma_input': 1.,
#             'grid_mean_predictor': {'type': 'cortex', 'input_dimensions': 2, 'hidden_layers': 0, 'hidden_features': 0,
#                                     'final_tanh': False},
#             'transfer_state_dict': 'd99820ef586bbebcfce36d7bc89877f3.pth.tar',
#             'init_transform_scale': 0.2,
#
#         }
#         print(model_fn)
#         print(model_config)
#         model_config_auto = dict(
#             gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
#             share_features={"type": "choice", "values": [True, False]},
#             share_transform={"type": "choice", "values": [True, False]},
#             #init_transform_scale={"type": "range", "bounds": [0.1, 0.7]},
#         )
#
#         trainer_fn = 'nnsysident.training.trainers.standard_trainer'
#         trainer_config = dict(detach_core=True)
#         trainer_config_auto = dict(
#         )
#
#         autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                              model_fn, model_config, model_config_auto,
#                              trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                              trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesianTransfer',
#                              total_trials=100)
#
#         best_parameters, _, _, _ = autobayes.run()


## Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

########      Full SXF 2d TRANSFER    ######################
#
# for neuron_n in [1000]:  # 1000,
#     for image_n in [200]:  # [ 50, 200, 500, 1000, 2500, 4399]
#         paths = ['data/static22564-2-12-preproc0.zip',
#                  'data/static22564-2-13-preproc0.zip',
#                  'data/static22564-3-8-preproc0.zip',
#                  'data/static22564-3-12-preproc0.zip']
#
# # for neuron_n in [5335]:  # 5335,
# #     for image_n in [50, 200]:  # [ 50, 200, 500, 1000, 2500, 4472]
# #         paths = ['data/static20457-5-9-preproc0.zip']
#
#
#         dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
#
#         dataset_config = dict(
#             paths=paths,
#             batch_size=64,
#             multi_match_n=neuron_n,
#             multi_match_base_seed=1,
#             image_n=image_n,
#             image_base_seed=1
#         )
#         #
#         # dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
#         #
#         # dataset_config = dict(
#         #     paths=paths,
#         #     batch_size=64,
#         #     neuron_n=neuron_n,
#         #     neuron_base_seed=1,
#         #     image_n=image_n,
#         #     image_base_seed=1
#         # )
#
#
#         dataset_config_auto = dict()
#         print(dataset_config)
#
#
#
#         model_fn = 'nnsysident.models.models.se2d_fullSXF'
#         model_config = {'gamma_input': 1.,
#                         "input_kern": 15,
#                         "hidden_kern": 13,
#                         'normalize': False,
#                         'transfer_state_dict': 'ed4a5d0bf35948b7ea65ee48bb5b5d93-4factorized.pth.tar',
#                         }
#         print(model_fn)
#         print(model_config)
#         model_config_auto = dict(
#             gamma_readout={"type": "range", "bounds": [0.0001, 0.5], "log_scale": True},
#             share_features={"type": "choice", "values": [True, False]},
#         )
#
#         trainer_fn = 'nnsysident.training.trainers.standard_trainer'
#         trainer_config = dict(detach_core=True)
#         trainer_config_auto = dict(
#         )
#
#         autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                              model_fn, model_config, model_config_auto,
#                              trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                              trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesianTransfer',
#                              total_trials=100)
#
#         best_parameters, _, _, _ = autobayes.run()


# ### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# ##########      Full Gaussian 2d - Different special cores     ######################
#
#
#
# comment = '11 datasets'
#
#
# dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
# dataset_config = {'paths': ['data/static22564-2-12-preproc0.zip',
#     'data/static22564-2-13-preproc0.zip',
#     'data/static22564-3-8-preproc0.zip',
#     'data/static22564-3-12-preproc0.zip',
#     'data/static22846-2-19-preproc0.zip',
#     'data/static22846-2-21-preproc0.zip',
#     'data/static22846-10-16-preproc0.zip',
#     'data/static23343-5-17-preproc0.zip',
#     'data/static23555-4-20-preproc0.zip',
#     'data/static23555-5-12-preproc0.zip',
#     'data/static23656-14-22-preproc0.zip'],
#    'batch_size': 64}
#
# dataset_config_auto = dict()
# print(dataset_config)
#
# model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
# model_config = {
#                 "init_mu_range": 0.55,
#                 "init_sigma": 0.4,
#                 'input_kern': 15,
#                 'hidden_kern': 13,
#                 'gamma_input': 1.,
#                 'grid_mean_predictor': {'type': 'cortex', 'input_dimensions': 2, 'hidden_layers': 0, 'hidden_features': 0, 'final_tanh': False},
# }
# print(model_fn)
# print(model_config)
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
# )
#
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict(detach_core=False)
# trainer_config_auto = dict(
#               )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
# best_parameters, _, _, _ = autobayes.run()
#
#
# model_config.update(best_parameters['model'])
# dataset_config.update(best_parameters['dataset'])
#
# Model().add_entry(model_fn=model_fn,
#                   model_config=model_config,
#                   model_fabrikant='kklurz',
#                   model_comment=comment)
# Dataset().add_entry(dataset_fn=dataset_fn,
#                     dataset_config=dataset_config,
#                     dataset_fabrikant='kklurz',
#                     dataset_comment=comment,
#                     skip_duplicates=True)


########################################################################################################################
#################################     ALAN DATA ######################################################################


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##########      Full Gaussian 2d ALAN    ######################

# paths = ['data/allen-agglomerate-NaturalScenesOracle-028f7ad795.zip']
#
# dataset_fn = 'nnsysident.datasets.mouse_loaders.mouse_allen_scene_loaders'
# dataset_config = dict(
#     paths=paths,
#     batch_size=64,
#     areas=('VISp',)
# )
#
# dataset_config_auto = dict()
# print(dataset_config)
#
# model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
# model_config = {"share_features":False,
#                 "share_transform":False,
#                 "init_mu_range": 0.55,
#                 "init_sigma": 0.4,
#                 'input_kern': 15,
#                 'hidden_kern': 13,
#                 'gamma_input': 1.,
#                 'grid_mean_predictor': {'type': 'cortex',
#                                         'input_dimensions': 2,
#                                         'hidden_layers': 0,
#                                         'hidden_features': 0,
#                                         'final_tanh': False},
#                 }
# print(model_fn)
# print(model_config)
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
#     init_transform_scale={"type": "range", "bounds": [0.1, 0.7]},
# )
#
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict(detach_core=True)
# trainer_config_auto = dict(
#               )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
# best_parameters, _, _, _ = autobayes.run()


## Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##########      Full Gaussian 2d TRANSFER ALAN   ######################


# paths = ['data/allen-agglomerate-NaturalScenesOracle-028f7ad795.zip']
#
# dataset_fn = 'nnsysident.datasets.mouse_loaders.mouse_allen_scene_loaders'
# dataset_config = dict(
#     paths=paths,
#     batch_size=64,
#     areas=('VISp',)
# )
#
#
# dataset_config_auto = dict()
# print(dataset_config)
#
# model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
# model_config = {"share_features":False,
#                 "share_transform":False,
#                 "init_mu_range": 0.55,
#                 "init_sigma": 0.4,
#                 'input_kern': 15,
#                 'hidden_kern': 13,
#                 'gamma_input': 1.,
#                 'grid_mean_predictor': {'type': 'cortex',
#                                         'input_dimensions': 2,
#                                         'hidden_layers': 0,
#                                         'hidden_features': 0,
#                                         'final_tanh': False},
#                 'transfer_state_dict': 'd99820ef586bbebcfce36d7bc89877f3-11gaussian.pth.tar',
#                 }
#
# print(model_fn)
# print(model_config)
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
#     init_transform_scale={"type": "range", "bounds": [0.1, 0.7]},
# )
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict(detach_core=True)
# trainer_config_auto = dict(
# )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesianTransfer',
#                      total_trials=100)
#
# best_parameters, _, _, _ = autobayes.run()


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


##########      Task driven     ######################

#
# paths = ['data/allen-agglomerate-NaturalScenesOracle-028f7ad795.zip']
#
# dataset_fn = 'nnsysident.datasets.mouse_loaders.mouse_allen_scene_loaders'
# dataset_config = dict(
#     paths=paths,
#     batch_size=64,
#     areas=('VISp',),
#     exclude=[],
# )
# dataset_config_auto = dict()
# print(dataset_config)
#
#
# model_fn = 'nnsysident.models.models.taskdriven_fullgaussian2d'
# model_config = {'tl_model_name': "vgg16",
#                 'layers': 8,
#                  'init_mu_range': 0.55,
#                  'init_sigma': 0.4,
#                  'share_features': False,
#                  'pretrained': True,
#                  'grid_mean_predictor': {'type': 'cortex',
#                   'input_dimensions': 2,
#                   'hidden_layers': 0,
#                   'hidden_features': 0,
#                   'final_tanh': False}}
#
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True}
# )
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict(detach_core=True)
# trainer_config_auto = dict(
#               )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
# best_parameters, _, _, _ = autobayes.run()


### Bayesian  -- Don't forget the schema!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


##########      Linear model     ######################

# paths = ['data/static20457-5-9-preproc0.zip']
#
# dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
# dataset_config = dict(
#     paths=paths,
#     batch_size=64,
#     neuron_n=5335,
#     image_n=4472,
# )
# dataset_config_auto = dict()
# print(dataset_config)
#
# model_fn = 'nnsysident.models.models.se2d_fullSXF'
# model_config = {
#                 'input_kern': 15,
#                 'hidden_kern': 13,
#                 'linear': True,
#                 'normalize': False,
#                 'share_features': False,
# }
#
# print(model_fn)
# print(model_config)
# model_config_auto = dict(
#     gamma_readout={"type": "range", "bounds": [1e-5, 1e3], "log_scale": True},
#     gamma_input={"type": "range", "bounds": [1e-3, 1e2], "log_scale": True},
# )
#
#
# trainer_fn = 'nnsysident.training.trainers.standard_trainer'
# trainer_config = dict(detach_core=False)
# trainer_config_auto = dict(
#               )
#
# autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
#                      model_fn, model_config, model_config_auto,
#                      trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
#                      trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=100)
#
# best_parameters, _, _, _ = autobayes.run()
#
#
# model_config.update(best_parameters['model'])
# dataset_config.update(best_parameters['dataset'])
#
# Model().add_entry(model_fn=model_fn,
#                   model_config=model_config,
#                   model_fabrikant='kklurz',
#                   model_comment='{}, linear model'.format(model_fn.split('.')[-1]))
# Dataset().add_entry(dataset_fn=dataset_fn,
#                     dataset_config=dataset_config,
#                     dataset_fabrikant='kklurz',
#                     dataset_comment='neuron_n={}, image_n={}'.format(dataset_config['neuron_n'], dataset_config['image_n']),
#                     skip_duplicates=True)


# OracleScore.populate(reserve_jobs=True)
# OracleScoreTransfer.populate(reserve_jobs=True)
