import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent import futures
from functools import partial
import tqdm 

np.random.seed(0)

class Writer:
    def __init__(self, outdir, type_name, start_idx=0,):
        self.outdir = Path(outdir)/ type_name
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.idx = start_idx
    def write(self, X, y):
        save_file = self.outdir / f'{self.idx}.npz'
        np.savez_compressed(save_file, X=X, y=y)
        self.idx += 1
    def start(self):
        print('Start writing to: ', self.outdir)

def write_to_file(writer, X, y):
    writer.start()
    try:
        for xi, yi in tqdm.tqdm(zip(X, y)):
            writer.write(xi, yi)
    except: return False
    return True

def train_test_split(N, val_fraction, test_fraction):
    """
    Input: 
    N: the size of the dataset
    val_fraction: the portion for validation set
    test_fraction: the portion for test set
    Output:
    Return the index for train, val, and test set
    """
    test_size = int(N * test_fraction)
    val_size = int(N * val_fraction)
    indices = np.random.permutation(N) 
    val_idx = indices[:val_size]
    test_idx = indices[val_size:val_size + test_size]
    train_idx = indices[val_size + test_size:]
    return [train_idx, val_idx, test_idx]
    
def resampling_data(car_model, in_dir, file_type, N_samples, attack_normal_ratio):
    """ 
    car_model: BMW, Tesla, Kia
    in_dir: directory for input data
    file_type: train, test, val
    N_samples: the size of total sampling data
    """
    in_dir = in_dir + '/{}/'  # to adapt with car_model
    in_path = Path(in_dir.format(car_model))
    classes = ['Normal', 'Fuzzy', 'Replay']
    def read_file(f):
        data = np.load(in_path / f)
        return data['X'], data['y']

    def sampling_data(d, indices):
        return d[indices]

    file_name = f'{file_type}_{{}}.npz'
    files = [file_name.format(c) for c in classes]
    data = list(map(lambda x: read_file(x), files))
    
    # Calculate the size of each class based on the attack/normal ratio
    class_distribution = np.array([1 - attack_normal_ratio, attack_normal_ratio / 2, attack_normal_ratio / 2])
    class_size = (N_samples * class_distribution).astype('int')
    # Sampling the indices according to the generated size
    indices = [np.arange(len(d[0])) for d in data]
    sampling_indices = [np.random.choice(idx, 
                                        size=size if size <= len(idx) else len(idx), 
                                        replace=False) 
                                        for idx, size in zip(indices, class_size)]
    # Take the data from sampling_indices
    X_subset = list(map(lambda p: sampling_data(p[0][0], p[1]), zip(data, sampling_indices)))
    y_subset = list(map(lambda p: sampling_data(p[0][1], p[1]), zip(data, sampling_indices)))
    return class_distribution, class_size, X_subset, y_subset