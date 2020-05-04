import os
import datajoint as dj
import pandas as pd

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['enable_python_native_blobs'] = True

name = "simdata"
dj.config['schema_name'] = f"konstantin_nnsysident_{name}"

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from nnsysident.tables.experiments import *

### Experiment

# experiment_name = 'SIM, Direct, se2d_fullgaussian2d, 0-0-3'
#
# TrainedModel.progress(Experiments.Restrictions & 'experiment_name="{}"'.format(experiment_name))
#
# TrainedModel.populate(Experiments.Restrictions & 'experiment_name="{}"'.format(experiment_name),
#                       reserve_jobs=True,
#                       order="random",)

### Bayesian

paths = ['data/static0-0-3-preproc0.zip']

dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
dataset_config = dict(
    paths=paths,
    batch_size=64,
    neuron_n=1000,
    neuron_base_seed=1,
    image_n=50,
    image_base_seed=1

)
dataset_config_auto = dict()
print(dataset_config)

model_fn = 'nnsysident.models.models.se2d_pointpooled'
model_config = {
}
model_config_auto = dict(
    gamma_input={"type": "range", "bounds": [1e-1, 1e3], "log_scale": True},
    gamma_readout={"type": "range", "bounds": [1e-3, 1e1], "log_scale": True},
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
                 dataset_comment='neuron_n={}, image_n={}'.format(dataset_config['neuron_n'], dataset_config['image_n']))

