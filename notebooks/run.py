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

### Experiment
#
# experiment_name = 'SIM, Direct, se2d_pointpooled, 0-0-3'
#
# TrainedModel.progress(Experiments.Restrictions & 'experiment_name="{}"'.format(experiment_name))
#
# TrainedModel.populate(Experiments.Restrictions & 'experiment_name="{}"'.format(experiment_name),
#                       reserve_jobs=True,
#                       order="random",)

### Transfer Experiment

# for experiment_name in ["SIM, core_transfer (best), se2d_fullgaussian2d, 0-0-3 -> 0-0-3",
#                         "SIM, core_transfer (best), se2d_pointpooled, 0-0-3 -> 0-0-3",
#                         "SIM, core_transfer (best), se2d_spatialxfeaturelinear, 0-0-3 -> 0-0-3"]:
#
#     TrainedModelTransfer.progress(ExperimentsTransfer.Restrictions & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModelTransfer.populate(ExperimentsTransfer.Restrictions & 'experiment_name="{}"'.format(experiment_name),
#                           reserve_jobs=True,
#                           order="random",)


### Bayesian

paths = ['data/static22564-2-12-preproc0.zip',
         'data/static22564-2-13-preproc0.zip',
         'data/static22564-3-8-preproc0.zip',
         'data/static22564-3-12-preproc0.zip']

dataset_fn = 'nnsysident.datasets.mouse_loaders.static_shared_loaders'
dataset_config = dict(
    paths=paths,
    batch_size=64,
    multi_match_n=3625,
    multi_match_base_seed=1,
    image_n=4399,
    image_base_seed=1

)
dataset_config_auto = dict()
print(dataset_config)

model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
model_config = {
}
model_config_auto = dict(
    gamma_input={"type": "range", "bounds": [1e-1, 1e3], "log_scale": True},
    gamma_readout={"type": "range", "bounds": [1e-5, 1e-1], "log_scale": True},

)


trainer_fn = 'nnsysident.training.trainers.standard_trainer'
trainer_config = dict()
trainer_config_auto = dict(
              )

autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
                     model_fn, model_config, model_config_auto,
                     trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
                     trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=50)

best_parameters, _, _, _ = autobayes.run()


Model().add_entry(model_fn=model_fn,
                  model_config=best_parameters['model'],
                  model_fabrikant='kklurz',
                  model_comment='{}, neuron_n={}, image_n={}'.format(model_fn.split('.')[-1], dataset_config['neuron_n'], dataset_config['image_n']))
Dataset().add_entry(dataset_fn=dataset_fn,
                    dataset_config=dataset_config,
                    dataset_fabrikant='kklurz',
                    dataset_comment='neuron_n={}, image_n={}'.format(dataset_config['neuron_n'], dataset_config['image_n']),
                    skip_duplicates=True)

