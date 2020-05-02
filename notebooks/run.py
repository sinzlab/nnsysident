import os
import datajoint as dj
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['enable_python_native_blobs'] = True

name = "simdata"
dj.config['schema_name'] = f"konstantin_nnsysident_{name}"

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *


paths = ['data/static0-0-3-preproc0.zip']

dataset_fn = 'nnsysident.datasets.mouse_loaders.static_loaders'
dataset_config = dict(
    paths=paths,
    batch_size=64,
    neuron_n=1000,
    neuron_base_seed=1,
    image_n=4000,
    image_base_seed=1

)
dataset_config_auto = dict()


model_fn = 'nnsysident.models.models.se2d_fullgaussian2d'
model_config = {
}
model_config_auto = dict(
    gamma_input={"type": "range", "bounds": [1e-2, 1e3], "log_scale": True},
    gamma_readout={"type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
)


trainer_fn = 'nnsysident.training.trainers.standard_trainer'
trainer_config = dict()
trainer_config_auto = dict(
              )

autobayes = Bayesian(dataset_fn, dataset_config, dataset_config_auto,
                     model_fn, model_config, model_config_auto,
                     trainer_fn, trainer_config, trainer_config_auto, architect="kklurz",
                     trained_model_table='nnsysident.tables.bayesian.TrainedModelBayesian', total_trials=200)

best_parameters, _, _, _ = autobayes.run()


Model().add_entry(model_fn=model_fn,
                  model_config=best_parameters['model'],
                  model_fabrikant='kklurz',
                  model_comment='neuron_n={}, image_n={}'.format(dataset_config['neuron_n'], dataset_config['image_n']))
Dataset().add_entry(dataset_fn=dataset_fn,
                  dataset_config=dataset_config,
                  dataset_fabrikant='kklurz',
                  dataset_comment='neuron_n={}, image_n={}'.format(dataset_config['neuron_n'], dataset_config['image_n']))
