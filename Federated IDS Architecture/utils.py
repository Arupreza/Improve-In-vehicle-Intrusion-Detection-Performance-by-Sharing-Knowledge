from typing import OrderedDict
from anyio import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import CANDataset

def get_parameters(model):
    """
    Return a list of parameters of a model
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, params):
    params_dicts = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dicts})
    model.load_state_dict(state_dict, strict=True)

def print_results(results, classes):
    print('\t' + '\t'.join(map(str, results.keys())))
    for idx, c in enumerate(classes):
        res = [round(results[k][idx], 4) for k in results.keys()]
        output = [c] + res
        print('\t'.join(map(str, output)))

def test_model(data_dir, model):
    transform = None
    test_dataset = CANDataset(root_dir=Path(data_dir), is_binary=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, 
                        pin_memory=True, sampler=None)
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    results = trainer.predict(model, dataloaders=test_loader)
    labels = np.concatenate([x['labels'] for x in results])
    preds = np.concatenate([x['preds'] for x in results])
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    prec, rec, f1 = precision_recall_fscore_support(labels, preds, average='binary')[:-1] 
    err = (fp + fn) / (tn + fp + fn + tp)
    far = fp / (tn + fp)
    return (err, far, prec, rec, f1)