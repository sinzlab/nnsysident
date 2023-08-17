import os
import tempfile

import datajoint as dj
import torch
from nnfabrik.builder import resolve_fn
from nnfabrik.utility.dj_helpers import make_hash
from nnfabrik.main import *
from nnfabrik.templates.trained_model import TrainedModelBase

if not "stores" in dj.config:
    dj.config["stores"] = {}
dj.config["stores"]["minio"] = {
    "protocol": "s3",
    "endpoint": os.environ["MINIO_ENDPOINT"],
    "bucket": "kklurzmodels",
    "location": "dj-store",
    "access_key": os.environ["MINIO_ACCESS_KEY"],
    "secret_key": os.environ["MINIO_SECRET_KEY"],
    "secure": True,
}

# create the context object
try:
    main = my_nnfabrik(os.environ["DJ_SCHEMA_NAME"], use_common_fabrikant=False)
except:
    raise ValueError(
        " ".join(
            [
                "No schema name has been specified.",
                "Specify it via",
                "os.environ['DJ_SCHEMA_NAME']='schema_name'",
            ]
        )
    )
# set some local variables such that the tables can be directly importable elsewhere
for key, val in main.__dict__.items():
    locals()[key] = val


@schema
class TrainedModel(TrainedModelBase):
    nnfabrik = main
    data_info_table = None
    table_comment = "Trained models"
    storage = "minio"

    @property
    def definition(self):
        definition = """
        # {table_comment}
        -> self().model_table
        -> self().dataset_table
        -> self().trainer_table
        -> self().seed_table
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # score
        train_loss:                        float        # train_loss
        validation_loss:                   float        # validation_loss
        test_loss:                         float        # test_loss
        train_correlation:                 float        # train_correlation
        validation_correlation:            float        # validation_correlation
        test_correlation:                  float        # test_correlation
        output:                            longblob     # trainer object's output
        ->[nullable] self().user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(
            table_comment=self.table_comment
        )
        return definition

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """
        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = self.user_table.get_current_user()
        seed = (self.seed_table & key).fetch1("seed")

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False, seed=seed)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)
        print("Finished training!")

        # save resulting model_state into a temporary file to be attached
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + ".pth.tar"
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key["score"] = score
            key["train_loss"] = output["best_model_stats"]["loss"]["train"]
            key["validation_loss"] = output["best_model_stats"]["loss"]["validation"]
            key["test_loss"] = output["best_model_stats"]["loss"]["test"]
            key["train_correlation"] = output["best_model_stats"]["correlation"]["train"]
            key["validation_correlation"] = output["best_model_stats"]["correlation"]["validation"]
            key["test_correlation"] = output["best_model_stats"]["correlation"]["test"]

            key["output"] = output
            key["fabrikant_name"] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key["comment"] = self.comment_delimitter.join(comments)
            print("Inserting in TrainedModel table...")
            self.insert1(key)

            key["model_state"] = filepath

            print("Inserting in ModelStorage table...")
            self.ModelStorage.insert1(key, ignore_extra_fields=True)



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
    -> Fabrikant.proj(experiment_fabrikant='fabrikant_name')
    experiment_comment='': varchar(2000)              # short description 
    experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    class Restrictions(dj.Part):
        definition = """
        # This table contains the corresponding hashes to filter out models which form the respective experiment
        -> master
        -> Dataset
        -> Trainer
        -> Model
        ---
        experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
        """

    def add_entry(
        self,
        experiment_name,
        experiment_fabrikant,
        experiment_comment,
        restrictions,
        skip_duplicates=False,
    ):
        self.insert1(
            dict(
                experiment_name=experiment_name,
                experiment_fabrikant=experiment_fabrikant,
                experiment_comment=experiment_comment,
            ),
            skip_duplicates=skip_duplicates,
        )

        restrictions = [{**{"experiment_name": experiment_name}, **res} for res in restrictions]
        self.Restrictions.insert(restrictions, skip_duplicates=skip_duplicates)
