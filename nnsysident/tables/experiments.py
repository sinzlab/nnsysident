import os
import datajoint as dj
import tempfile
import torch

from nnfabrik.template import TrainedModelBase
from nnfabrik.main import *
from nnfabrik.builder import resolve_fn

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
class Transfer(dj.Manual):
    definition = """
    # This table contains different ways of conducting transfer experiments with models  
    
    transfer_fn:                     varchar(64)    # name of the transfer function
    transfer_hash:                   varchar(64)    # hash of the configuration object
    ---
    transfer_config:                 longblob       # transfer configuration object
    -> Fabrikant.proj(transfer_fabrikant='fabrikant_name')
    transfer_comment='' :            varchar(256)    # short description
    transfer_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    def add_entry(self, transfer_fn, transfer_config, transfer_fabrikant=None, transfer_comment='', skip_duplicates=False):
        """
        Add a new entry to the transfer.

        Args:
            transfer_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `transfer` subpackage.
            transfer_config (dict) - Python dictionary containing keyword arguments for the transfer_fn
            transfer_fabrikant (string) - The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based on the database user name for the existing connection.
            transfer_comment - Optional comment for the entry.
            skip_duplicates - If True, no error is thrown when a duplicate entry (i.e. entry with same transfer_fn and transfer_config) is found.

        Returns:
            key - key in the table corresponding to the entry.
        """
        try:
            resolve_fn(transfer_fn, 'transfer')
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + '\nTable entry rejected')
            return

        if transfer_fabrikant is None:
            transfer_fabrikant = Fabrikant.get_current_user()

        transfer_hash = make_hash(transfer_config)
        key = dict(transfer_fn=transfer_fn, transfer_hash=transfer_hash, transfer_config=transfer_config,
                   transfer_fabrikant=transfer_fabrikant, transfer_comment=transfer_comment)

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn('Corresponding entry found. Skipping...')
                key = (self & (existing)).fetch1()
            else:
                raise ValueError('Corresponding entry already exists')
        else:
            self.insert1(key)

        return key



@schema
class TrainedModelTransfer(TrainedModelBase):
    model_table = Model
    dataset_table = Dataset
    trainer_table = Trainer
    seed_table = Seed
    user_table = Fabrikant
    transfer_table = Transfer

    # delimitter to use when concatenating comments from model, dataset, and trainer tables
    comment_delimitter = '.'

    # table level comment
    table_comment = "Trained models from transfer experiments"

    class ModelStorage(TrainedModelBase.ModelStorage):
        storage = "minio_models"

    @property
    def definition(self):
        definition = """
        # {table_comment}
        -> self.model_table
        -> self.dataset_table
        -> self.trainer_table
        -> self.seed_table
        -> self.transfer_table
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self.user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(table_comment=self.table_comment)
        return definition

    def make(self, key):
        print('Populating key: {}'.format(key))
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """
        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = self.user_table.get_current_user()
        seed = (self.seed_table & key).fetch1('seed')

        # extract the transfer_fn and transfer_hash so the key fits nnFabrik standards
        transfer_fn = key.pop('transfer_fn')
        transfer_hash = key.pop('transfer_hash')

        transfer_config = (Transfer & 'transfer_hash="{}"'.format(transfer_hash)).fetch1('transfer_config')
        trainer_config = (Trainer & 'trainer_hash="{}"'.format(key['trainer_hash'])).fetch1('trainer_config')

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False, seed=seed)

        # Conduct the transfer defined by the transfer function
        transfer_function = resolve_fn(transfer_fn, default_base=None)
        transfer_function(model=model, trained_model_table=TrainedModel, trainer_config=trainer_config, **transfer_config)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + '.pth.tar'
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key['score'] = score
            key['output'] = output
            key['fabrikant_name'] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key['comment'] = self.comment_delimitter.join(comments)
            key['transfer_fn'] = transfer_fn
            key['transfer_hash'] = transfer_hash
            self.insert1(key)

            key['model_state'] = filepath

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
    -> Fabrikant.proj()
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


