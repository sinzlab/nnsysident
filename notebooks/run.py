import os
import datajoint as dj
import pandas as pd

dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASS"]
dj.config["enable_python_native_blobs"] = True

from nnfabrik.main import my_nnfabrik
name = "playground"
my_nnfabrik(f"konstantin_nnsysident_{name}", context=locals())

from nnfabrik.utility.hypersearch import Bayesian
from nnfabrik.main import *
from nnsysident.tables.experiments import *
from nnsysident.tables.scoring import OracleScore, OracleScoreTransfer, R2erScore, R2erScoreTransfer, FeveScore, FeveScoreTransfer



### Experiment

# for experiment_name in ['Real, Direct, se2d_fullgaussian2d, 20457-5-9, AnscombeLoss']:
#
#     TrainedModel.progress(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name))
#
#     TrainedModel.populate(Experiments.Restrictions & 'seed in (1,2,3,4,5)' & 'experiment_name="{}"'.format(experiment_name),
#                           reserve_jobs=True,
#                           order="random",)
#
#

OracleScore.populate(reserve_jobs=True)
OracleScoreTransfer.populate(reserve_jobs=True)
