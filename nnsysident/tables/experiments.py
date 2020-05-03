import os
import datajoint as dj

from nnfabrik.template import TrainedModelBase


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
    definition = """
    # Table to keep track of collections of trained networks that form an experiment. 
    # Instructions: 
    # 1) Insert all combinations of dataset, model and trainer together with a common experiment name in this table. 
    # 2) Populate the TrainedModel table by restricting it with this table and the experiment name.
    # 3) After training, join this table with TrainedModel and restrict by experiment name to get your results
       
    -> Dataset
    -> Trainer
    -> Model
    experiment_name: varchar(100)                     # name of experiment
    ---
    -> Fabrikant.proj()
    experiment_comment='': varchar(2000)              # short description 
    experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """
