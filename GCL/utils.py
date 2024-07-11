from typing import *
import os
import torch

import random
import numpy as np


def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def build_dgl_graph(edge_index: torch.Tensor) -> 'dgl.DGLGraph':    # noqa
    import warnings
    warnings.warn("This method may not be maintained in future versions", FutureWarning)
    try:
        import dgl
    except ImportError:
        raise ImportError("DGL is not installed. Please install it.")
    
    row, col = edge_index
    return dgl.graph((row, col))


def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res
