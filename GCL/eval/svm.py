from typing import Callable, Optional, Dict
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import BaseCrossValidator

from GCL.eval import BaseSKLearnEvaluator


class SVMEvaluator(BaseSKLearnEvaluator):
    """
    Evaluate using the sklearn SVM classifier.

    Parameters:
        metrics (Dict[str, Callable]): The metrics to evaluate in a dictionary
            with metric names as keys and callables as values.
        split (BaseCrossValidator): The sklearn cross-validator to split the data.
        linear (bool): Whether to use linear SVM. (default: :obj:`True`)
        params (Dict, optional): Other parameters for the SVM model.
            See sklearn `SVC
            <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
            for details. (default: :obj:`None`)
        param_grid (List[Dict], optional): The parameter grid for the grid search. (default: :obj:`None`)
        grid_search_scoring (Dict[str, Callable], optional):
            If :obj:`param_grid` is given, provide metrics in grid search.
            If multiple metrics are given, the first one will be used to retrain the best model.
            (default: :obj:`None`)
        grid_search_params (Dict, optional): If :obj:`param_grid` is given, further pass the parameters
            for the sklearn grid search cross-validator. See sklearn `GridSearchCV
            <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
            for details. (default: :obj:`None`)
    """
    def __init__(
            self, metrics: Dict[str, Callable], split: BaseCrossValidator,
            linear=True, params: Optional[Dict] = None, param_grid: Optional[Dict] = None,
            grid_search_scoring: Optional[Dict[str, Callable]] = None,
            grid_search_params: Optional[Dict] = None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        super(SVMEvaluator, self).__init__(
            evaluator=self.evaluator, metrics=metrics, split=split, params=params,
            param_grid=param_grid, grid_search_scoring=grid_search_scoring, grid_search_params=grid_search_params)
