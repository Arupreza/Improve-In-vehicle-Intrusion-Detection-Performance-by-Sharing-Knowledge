from argparse import ArgumentParser
from os import PathLike
from re import I
import flwr as fl
from central_ids import IDS
import pytorch_lightning as pl
import torch
from utils import get_parameters, set_parameters

DEFAULT_SERVER_ADDRESS = "[::]:8080"

class DistributedIDS(fl.client.NumPyClient):
    def __init__(self, **kwargs):
        self.model = IDS(**kwargs)
        self.args = kwargs

    def get_parameters(self, config):
        return get_parameters(self.model) 

    def set_parameters(self, params):
        set_parameters(self.model, params)
    
    def get_hyperparam(self, config):
        for k, v in config.items():
            if k in self.args:
                self.args[k] = v
    
    def fit(self, params, config):
        self.set_parameters(params)
        self.model.update_args(config)
        trainer = pl.Trainer(max_epochs=config['epochs'], 
                            accelerator=self.args['device'],
                            check_val_every_n_epoch=self.args['val_freq'],)
        trainer.fit(self.model)
        trained_params = get_parameters(self.model)
        num_examples = self.model.get_train_size()
        return trained_params, num_examples, {}
        
    def evaluate(self, params, config):
        """
        Evaluate on local test set
        """
        print('Local evaluation')
        set_parameters(self.model, params)
        trainer = pl.Trainer(accelerator=self.args['device'])
        print('Initalize: DONE')
        results = trainer.validate(self.model)
        f1 = results[0]['val_f1']
        loss = results[0]['val_loss']
        num_examples = self.model.get_val_size()
        return loss, num_examples, {"val_loss": loss, "val_f1": f1}


def argument_paser():
    parser = ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    # parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--C", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = argument_paser()
    client = DistributedIDS(**args)
    fl.client.start_numpy_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)