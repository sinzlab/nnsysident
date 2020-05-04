import os
import datajoint as dj

from nnfabrik.template import TrainedModelBase
from nnfabrik.main import *


if not 'stores' in dj.config:
    dj.config['stores'] = {}
dj.config["stores"]["minio_models"] = {
    "protocol": "s3",
    "endpoint": os.environ["MINIO_ENDPOINT"],
    "bucket": "kklurzmodels",
    "location": "dj-store",
    "access_key": os.environ["MINIO_ACCESS_KEY"],
    "secret_key": os.environ["MINIO_SECRET_KEY"],
}

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))

@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"

    class ModelStorage(TrainedModelBase.ModelStorage):
        storage = "minio_models"


@schema
class Experiments(dj.Manual):
    # Table to keep track of collections of trained networks that form an experiment.
    # Instructions:
    # 1) Make an entry in Experiments with an experiment name and description
    # 2) Insert all combinations of dataset, model and trainer for this experiment name in Experiments.Restrictions.
    # 2) Populate the TrainedModel table by restricting it with Experiments.Restrictions and the experiment name.
    # 3) After training, join this table with TrainedModel and restrict by experiment name to get your results
    definition = """
    # This table contains the experiments and their descriptions
    experiment_name: varchar(100)                     # name of experiment
    ---
    -> Fabrikant.proj()
    experiment_comment='': varchar(2000)              # short description 
    experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """
    class Restrictions(dj.Part):
        definition="""
        # This table contains the corresponding hashes to filter out models which form the respective experiment
        -> master
        -> Dataset
        -> Trainer
        -> Model
        ---
        experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
        """
