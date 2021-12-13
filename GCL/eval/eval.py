import torch
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict, Optional
from torch_geometric.data import Data
from sklearn.model_selection import GridSearchCV, BaseCrossValidator


def get_split(
        num_samples: int, num_splits: int = 1,
        train_ratio: float = 0.1, test_ratio: float = 0.8) -> Union[Dict, List[Dict]]:
    """
    Generate split indices for train, test, and validation sets.

    Args:
        num_samples (int): The size of the dataset.
        num_splits (int, optional): The number of splits to generate. (default: :obj:`1`)
        train_ratio (float, optional): The ratio of the train set. (default: :obj:`0.1`)
        test_ratio (float, optional): The ratio of the test set. (default: :obj:`0.8`)

    Returns:
        Union(Dict, List[Dict]): A dictionary of split indices or a list of dictionaries of split indices.

    Examples:
        >>> get_split(10, num_splits=1, train_ratio=0.5, test_ratio=0.4)
        [{'train': [3, 4, 0, 1, 2], 'test': [5, 7, 6, 8], 'valid': [9]}]
    """
    assert train_ratio + test_ratio < 1

    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    out = []
    for i in range(num_splits):
        indices = torch.randperm(num_samples)
        out.append({
            'train': indices[:train_size],
            'valid': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        })
    return out if num_splits > 1 else out[0]


def from_PyG_split(data: Data) -> Union[Dict, List[Dict]]:
    """
    Convert PyG split indices for train, test, and validation sets.

    Args:
        data (`Data`): The PyG data object.

    Returns:
        Union[Dict, List[Dict]]: A dictionary of split indices.

    Raises:
        ValueError: If the data object does not have the split indices.
    """
    if any([mask is None for mask in [data.train_mask, data.test_mask, data.val_mask]]):
        raise ValueError('The data object does not have the split indices.')
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)

    if data.train_mask.dim() == 1:
        return {
            'train': indices[data.train_mask],
            'valid': indices[data.val_mask],
            'test': indices[data.test_mask]
        }
    else:
        out = []
        for i in range(data.train_mask.size(1)):
            out_dict = {}
            for mask in ['train_mask', 'val_mask', 'test_mask']:
                if data[mask].dim() == 1:
                    # Datasets like WikiCS have only one split for the test set.
                    out_dict[mask[:-5]] = indices[data[mask]]
                else:
                    out_dict[mask[:-5]] = indices[data[mask][:, i]]
            out.append(out_dict)
        return out


class BaseEvaluator(ABC):
    """
    Base class for trainable (e.g., logistic regression) evaluation.

    Args:
        split (Union[Dict, List[Dict]]): The split indices.
        metric (Union[Callable, List[Callable]]): The metric(s) to evaluate.
        stop_metric (Union[None, Callable, int], optional): The metric(s) to stop training.
            It could be a callable function, or an integer specifying the index of the :obj:`metric`.
            If set to :obj:`None`, the stopping metric will be set to the first in :obj:`metric`.
             (default: :obj:`None`)
        cv (Optional[BaseCrossValidator], optional): The sklearn cross-validator. (default: :obj:`None`)
    """

    def __init__(
            self, split: Union[Dict, List[Dict]],
            metric: Union[Callable, List[Callable]], stop_metric: Union[None, Callable, int] = None,
            cv: Optional[BaseCrossValidator] = None):
        self.cv = cv
        self.split = split
        self.metric = metric
        if cv is None and stop_metric is None:
            stop_metric = 0
        if callable(metric):
            metric = [metric]
        if isinstance(stop_metric, int):
            if isinstance(metric, list):
                self.stop_metric = metric[stop_metric]
            else:
                raise ValueError
        else:
            self.stop_metric = stop_metric

    @abstractmethod
    def evaluate(self, x: Union[torch.FloatTensor, np.ndarray], y: Union[torch.LongTensor, np.ndarray]) -> Dict:
        raise NotImplementedError

    def __call__(self, x: Union[torch.FloatTensor, np.ndarray], y: Union[torch.LongTensor, np.ndarray]) -> Dict:
        result = self.evaluate(x, y)
        return result


class BaseSKLearnEvaluator:
    def __init__(
            self, evaluator: Callable, metric: Union[Callable, List[Callable]], split: BaseCrossValidator,
            param_grid: Optional[Dict] = None, search_cv: Optional[Callable] = None, refit: Optional[str] = None):
        self.evaluator = evaluator
        self.param_grid = param_grid
        self.split = split
        self.search_cv = search_cv
        self.refit = refit
        if callable(metric):
            metric = [metric]
        self.metric = metric

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        results = []
        for train_idx, test_idx in self.split.split(x, y):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            if self.param_grid is not None:
                classifier = GridSearchCV(
                    self.evaluator, param_grid=self.param_grid, cv=self.search_cv,
                    scoring=self.metric, refit=self.refit,
                    verbose=0, return_train_score=False)
            else:
                classifier = self.evaluator
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            results.append({metric.__name__: metric(y_test, y_pred) for metric in self.metric})

        results = pd.DataFrame.from_dict(results)
        return results.agg(['mean', 'std']).to_dict()
