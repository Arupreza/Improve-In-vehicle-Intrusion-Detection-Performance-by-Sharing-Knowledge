import pandas as pd
import numpy as np
from cmath import nan
import pywt
from pandarallel import pandarallel
from functools import partial
import torch

def convert_wavelet(signal, waveletname):
    N = len(signal)
    scales = np.arange(1, N + 1)
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname)
    coefficients = np.expand_dims(coefficients, axis=0)
    return coefficients

def create_wavelet_feature(sample, waveletname):
    wavelet_matrices = [convert_wavelet(signal, waveletname=waveletname) for signal in sample[:-1]]
    wavelet_features = np.concatenate(wavelet_matrices, axis=0)
    return np.expand_dims(wavelet_features, axis=0) 

def work(attack, data_dir, wavelet_family):
    # Read data time sequence
    data = torch.load(f'{data_dir}/time_sequence/{attack}.pt')
    data = [d.numpy().tolist() for d in data]
    cols = ['ID'] + [f'Data{x}' for x in range(8)] + ['Label']
    df = pd.DataFrame(dict(zip(cols, data)))
    # Convert wavelet
    partial_wavelet_feature = partial(create_wavelet_feature, waveletname=wavelet_family)
    X = df.parallel_apply(partial_wavelet_feature, axis=1)
    y = df.Label
    X_numpy = np.concatenate(X).astype(np.float16)
    y_numpy = y.to_numpy().astype(np.uint8)
    normal_indices = np.where(y_numpy == 0)
    attack_indices = np.where(~(y_numpy == 0)) 
    # Save normal 
    save_file = f'{data_dir}/wavelet/{wavelet_family}/Normal_{attack}.npz'
    print('Saving to: ', save_file)
    np.savez_compressed(save_file, X=X_numpy[normal_indices], y=y_numpy[normal_indices])
    # Save attack
    save_file = f'{data_dir}/wavelet/{wavelet_family}/{attack}.npz'
    print('Saving to: ', save_file)
    np.savez_compressed(save_file, X=X_numpy[attack_indices], y=y_numpy[attack_indices])

if __name__ == '__main__':
    attacks = ['DoS', 'Fuzzy', 'gear', 'RPM']
    data_dir = '../Data/CHD_w29_s14_ID_Data/'
    pandarallel.initialize(progress_bar=True)
    wavelet_family = 'mexh'
    for a in attacks[1:]:
        work(a, data_dir, wavelet_family)