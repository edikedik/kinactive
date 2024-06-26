"""
Models' interface and creation pipeline.
"""
from __future__ import annotations

import logging
import operator as op
import re
import typing as t
import warnings
from abc import ABCMeta, abstractmethod
from collections import abc
from itertools import chain
from statistics import mean

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import numpy as np
    import optuna
    import pandas as pd

    from eBoruta import eBoruta, Dataset, TrialData, Features
    from more_itertools import unique_everseen
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from toolz import curry
    from tqdm.auto import tqdm
    from xgboost import XGBClassifier, XGBRegressor

    from kinactive.config import ColNames, DFG_MAP_REV

PDB_PATTERN = re.compile(r"\((\w{4}):\w+\|")
LOGGER = logging.getLogger(__name__)


class ModelBase(metaclass=ABCMeta):
    """
    An abstract base class for model objects.
    """

    @property
    @abstractmethod
    def features(self) -> abc.Sequence[str]:
        """
        :return: A sequence of features available for the model.
        """

    @property
    @abstractmethod
    def targets(self) -> abc.Sequence[str]:
        """
        :return: A sequence of target variables.
        """

    @abstractmethod
    def reinit_model(self):
        """
        Reinitialize model.
        """

    @abstractmethod
    def train(self, df: pd.DataFrame):
        """
        Train the model on entirety of the provided data.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        """
        Make predictions from the provided data.
        """

    @abstractmethod
    def cv(self, df: pd.DataFrame, n: int) -> float:
        """
        Cross-validate the model.

        :param df: Data to use for training/testing.
        :param n: The number of CV folds.
        :return: A performance estimate aggregated across testing folds.
        """

    @abstractmethod
    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate fold indices from the provided data.

        :param df: DataFrame with predictors.
        :param n: The number of folds.
        :return: An iterator over tuples with train and test boolean indices
            allowing to select train and test observations from `df`.
        """

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        """
        Score the model.

        :class:`KinactiveClassifier` uses :func:`f1_score`.

        :class:`KinactiveRegressor` uses :func:`r2_score`.

        :param df: Data to predict from.
        :return: A single number -- model's performance estimate (the higher
            the better).
        """


class ObjectiveFn(t.Protocol):
    """
    An objective function type.
    """

    def __call__(
        self, trial: optuna.Trial, df: pd.DataFrame, model: ModelBase, n_cv: int
    ) -> float:
        ...


class ModelT(t.Protocol):
    """
    A minimalistic model interface.
    """

    def fit(
        self, x: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series, **kwargs
    ) -> ModelT:
        """
        Fit the model
        """

    def predict(self, x: pd.DataFrame | np.ndarray, **kwargs) -> np.ndarray:
        """Predict the results."""

    def predict_proba(self, x: pd.DataFrame | np.ndarray, **kwargs) -> np.ndarray:
        """Predict classes' probabilities."""


def xgb_objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model: KinactiveModel,
    n_cv: int = 5,
    use_early_stopping: bool = False,
) -> float:
    """
    A default objective function for XGB models. It uses the following setup::

        learning_rate:     [0, 1]
        max_depth:         [4, 16]
        gamma:             [0.0, 10.0]
        reg_lambda:        [0.0, 10.0]
        reg_alpha:         [0.0, 10.0]
        colsample_bytree:  [0.4, 1.0]
        colsample_bylevel: [0.4, 1.0]

    Additionally, for the ``XGBclassifier`` it adds::

        scale_pos_weight:  [0.0, 10.0]

    After the parameters are sampled, they are combined with the existing model
    parameters via ``{**model.params, **params}``. Then, the model is instantiated
    with the new parameters and cross-validated using :meth:`KinactiveModel.cv`.

    :param trial: A trial instance used dynamically by optuna. Leave as is.
    :param df: A dataset used to fit and test the model.
    :param model: The model to optimize the params for.
    :param n_cv: The number of CV folds to derive the score.
    :param use_early_stopping: Passed to the ``model``.
    :return: The cross-validated score.
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0, 1),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "gamma": trial.suggest_float("gamma", 0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
    }
    if isinstance(model, KinactiveClassifier):
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.0, 10.0)
    if not use_early_stopping:
        params["n_estimators"] = trial.suggest_int("max_depth", 10, 1000)

    # callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_logloss')
    params = {**model.params, **params}
    model = model.__class__(
        model.model,
        model.targets,
        model.features,
        params,
        use_early_stopping,
        cv_col=model.cv_col,
        weight_col=model.weight_col,
    )
    return model.cv(df, n_cv)


def lr_objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model: KinactiveModel,
    n_cv: int = 5,
    use_early_stopping: bool = False,
) -> float:
    """
    A default objective function for the logistic regression model.

    It optimizes the following params::

        C: [0.0, 1.0]
        class_weight: [None, "balanced"]
        solver: ["newton-cg", "sag", "saga", "lbfgs"]
        multi_class: ["auto", "ovr", "multinomial"]

    If ``solver == "saga"``, it encodes "l2" as the ``penalty`` parameters.
    Otherwise, it chooses between "l1", "l2", and "elasticnet".
    If the latter is chosen, it adds samples the ``l1_ratio`` parameter between
    zero and one.

    The options ``max_iter`` and ``n_jobs`` are hard-coded to 1000 and -1.

    After sampling, the process is identical to the :func:`xgb_objective`.

    :param trial: A trial instance used dynamically by optuna. Leave as is.
    :param df: A dataset used to fit and test the model.
    :param model: The model to optimize the params for.
    :param n_cv: The number of CV folds to derive the score.
    :param use_early_stopping: Passed to the ``model``.
    :return: The cross-validated score.
    """
    params = {
        "C": trial.suggest_float("C", 0, 1),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "solver": trial.suggest_categorical(
            "solver", ["newton-cg", "sag", "saga", "lbfgs"]
        ),
        "multi_class": trial.suggest_categorical(
            "multi_class", ["auto", "ovr", "multinomial"]
        ),
        "max_iter": 1000,
        # "n_jobs": -1,
    }
    if params["solver"] in ["newton-cg", "sag", "lbfgs"]:
        params["penalty"] = "l2"
    else:
        params["penalty"] = trial.suggest_categorical(
            "penalty", ["l1", "l2", "elasticnet"]
        )

    if params["penalty"] == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)

    params = {**model.params, **params}
    model = model.__class__(
        model.model,
        model.targets,
        model.features,
        params,
        use_early_stopping,
        cv_col=model.cv_col,
        weight_col=model.weight_col,
    )
    score = model.cv(df, n_cv)
    return score


def rf_objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model: KinactiveModel,
    n_cv: int = 5,
    use_early_stopping: bool = False,
) -> float:
    """
    A default objective function for random forests.

    :param trial: A trial instance used dynamically by optuna. Leave as is.
    :param df: A dataset used to fit and test the model.
    :param model: The model to optimize the params for.
    :param n_cv: The number of CV folds to derive the score.
    :param use_early_stopping: Passed to the ``model``.
    :return: The cross-validated score.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 1.0),
    }
    if isinstance(model, KinactiveClassifier):
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        )

    params = {**model.params, **params}
    model = model.__class__(
        model.model,
        model.targets,
        model.features,
        params,
        use_early_stopping,
        cv_col=model.cv_col,
        weight_col=model.weight_col,
    )
    score = model.cv(df, n_cv)
    return score


class KinactiveModel(ModelBase, metaclass=ABCMeta):
    """
    An interface wrapper around the ML algorithm.

    Its methods operate on a ``DataFrame``, applying stored :meth:`features`
    and :meth:`targets` to obtain necessary variables.

    .. seealso::

        :func:`make` -- a model's creation pipeline.

    """

    def __init__(
        self,
        model: ModelT,
        targets: abc.Iterable[str],
        features: abc.Iterable[str] = (),
        params: dict[str, t.Any] | None = None,
        use_early_stopping: bool = False,
        selector: eBoruta | None = None,
        cv_col: str = "ObjectID",
        weight_col: str | None = None,
        score_fn: abc.Callable[[abc.Sequence, abc.Sequence], float] | None = None
    ):
        """
        :param model: A model defining ``fit`` and ``predict`` methods.
            :meth:`select_params` assumes it to be either XGBoost or
            ``LogisticRegressionClassifier`.
        :param targets: Target variables' names.
        :param features: Feature variables' names.
        :param params: Initial parameters for the model.
        :param cv_col: A col used to generate non-overlapping CV folds.
        :param weight_col: A col pointing to sample weights.
        :param use_early_stopping: If ``True``, and the model is either
            ``XGBClassifier`` or ``XGBRegressor``, the :meth:`train` will
            split the provided dataset into training and evaluation parts
            and use the evaluation part to monitor the loss function and stop
            adding new trees (thus, finish training), if the loss didn't improve
            for a number of consecutive steps. The number of early stopping
            rounds should be provided in ``params``.
        :param selector: The feature selector to use in :meth:`select_params`.
        """
        if not isinstance(features, list):
            features = list(features)
        if not isinstance(targets, list):
            targets = list(targets)
        self._features = features
        self._targets = targets
        self._model = model
        #: A column pointing to values used to generate CV folds.
        self.cv_col = cv_col
        #: A column pointing to sample weights.
        self.weight_col = weight_col
        #: Model's parameters.
        self.params = params or {}
        #: Use early stopping via eval set.
        self.use_early_stopping = use_early_stopping
        #: eBoruta instance
        self.selector = selector
        #: Custom scoring function
        self.score_fn = score_fn

    @property
    def model(self):
        """
        :return: Current model instance.
        """
        return self._model

    @property
    def features(self) -> list[str]:
        """
        :return: A list of features used to train the model.
        """

        return self._features

    @property
    def targets(self) -> list[str]:
        return self._targets

    def reinit_model(self):
        if isinstance(self.model, type):
            self._model = self._model(**self.params)
        else:
            self._model = self._model.__class__(**self.params)

    def train(self, df: pd.DataFrame):
        if self.use_early_stopping and isinstance(
            self.model, (XGBClassifier, XGBRegressor)
        ):
            train_idx, eval_idx = next(self.generate_fold_idx(df, 10))
            train_df, train_ys, train_ws = _get_xy(
                df[train_idx], self.features, self.targets, self.weight_col
            )
            eval_df, eval_ys, eval_ws = _get_xy(
                df[eval_idx], self.features, self.targets, self.weight_col
            )
            self._model.fit(
                train_df,
                np.squeeze(train_ys.values),
                sample_weight=train_ws,
                eval_set=[(eval_df, np.squeeze(eval_ys.values))],
                sample_weight_eval_set=[eval_ws],
                verbose=0,
            )
        else:
            assert (
                "early_stopping_rounds" not in self.params
            ), "Must not have early stopping params if `use_early_stopping` is `False`"
            xs, ys, ws = _get_xy(df, self.features, self.targets, self.weight_col)
            assert ys is not None, f"failed finding target variables {self.targets}"
            if isinstance(self.model, (XGBClassifier, XGBRegressor)):
                self._model.fit(
                    xs.values, np.squeeze(ys.values), sample_weight=ws, verbose=0
                )
            else:
                self._model.fit(xs.values, np.squeeze(ys.values), sample_weight=ws)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = _apply_selection(df, self.features)
        return self._model.predict(df.values)

    def cv(
            self, df: pd.DataFrame, n: int, verbose: bool = False, scores: bool = False
    ) -> float:
        return _cross_validate(self, df, n, verbose, scores)

    def cv_pred(
        self, df: pd.DataFrame, n: int, verbose: bool = False, scores: bool = False
    ) -> tuple[float | list[float], pd.DataFrame]:
        """
        Cross-validate the score and predict the data in test folds.

        :param df: Input data with features and target columns.
        :param n: The number of CV folds to use.
        :param verbose: Output progress bar.
        :return: A tuple with score and a copy of the supplied dataframe with
            fold assignment and model prediction columns added.
        """
        return _cross_validate_and_predict(self, df, n, verbose, scores)

    def select_params(
        self,
        df: pd.DataFrame,
        n_trials: int,
        n_cv: int = 5,
        direction: str = "maximize",
        early_stopping_rounds: int = 0,
    ) -> optuna.Study:
        """
        Optimize hyperparameters.

        :param df: Input data with features and target columns.
        :param n_trials: The number of optimization rounds.
        :param n_cv: The number of CV folds to use within the objective.
        :param direction: "maximize" or "minimize" the objective.
        :param early_stopping_rounds: The number of early stopping rounds to use.
            Zero means no early stopping.
        :return: The ``Study`` instance from optuna.
        """
        if isinstance(self.model, (XGBClassifier, XGBRegressor)):
            obj = xgb_objective
        elif isinstance(self.model, RandomForestClassifier):
            obj = rf_objective
        elif isinstance(self.model, LogisticRegression):
            obj = lr_objective
        else:
            raise TypeError("Unsupported model type")

        objective = curry(obj)(
            df=df, model=self, n_cv=n_cv, use_early_stopping=self.use_early_stopping
        )
        study = optuna.create_study(direction=direction)
        cb = (
            [EarlyStoppingCallback(early_stopping_rounds, direction=direction)]
            if early_stopping_rounds
            else None
        )
        study.optimize(objective, n_trials=n_trials, callbacks=cb)
        self.params = {**self.params, **study.best_params}
        self.reinit_model()
        return study

    def select_features(self, df: pd.DataFrame, n_folds: int = 10, **kwargs) -> eBoruta:
        """
        Select important features and store the selection to :meth:`features`.

        :param df: A dataframe with features and targets.
        :param kwargs: Passed to the :attr:`selector`.
        :param n_folds: A number of CV folds to assess performance during
            evaluation for early stopping XGBoost callback.
        :return: The ``selector.fit()`` output.
        """
        if self.selector is None:
            self.selector = eBoruta(**kwargs)
        df_x, df_y, ws = _get_xy(df, self.features, self.targets, self.weight_col)
        assert df_y is not None
        self.reinit_model()
        if "early_stopping_rounds" in self.params and isinstance(
            self.model, (XGBClassifier, XGBRegressor)
        ):
            obj_ids = df[self.cv_col].values
            target = None
            if isinstance(self.model, XGBClassifier):
                target = np.squeeze(df[self.targets].values)
            callbacks = [EvalSetSupplier(obj_ids, target, n_folds)]
        else:
            callbacks = None
        res = self.selector.fit(
            df_x,
            np.squeeze(df_y.values),
            ws,
            model=self.model,
            callbacks_trial_start=callbacks,
        )
        self._features = list(res.features_.accepted)
        self.reinit_model()
        return res

    def rank_features(self, features: abc.Sequence[str] | None, **kwargs):
        """
        Rank features using ``selector.rank()``.

        :param features: A sequence of features. If not provided, will use
            :meth:`features`.
        :param kwargs: Passed to ``selector.rank()``.
        :return: A table with ranked features.
        """
        if self.selector is None:
            raise ValueError(
                "No selector instance present. Call `select_features` first"
            )
        features = features or self.features
        return self.selector.rank(features, **kwargs)


class KinactiveClassifier(KinactiveModel):
    """
    A model wrapper for classification objective.
    """

    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        return _generate_stratified_fold_idx(
            df[self.cv_col].values, np.squeeze(df[self.targets].values), n
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict classes' probabilities.

        :param df: A tabular dataset with features and target columns.
        :return: The array of predicted probabilities. Its shape depends on the
            number of targets and classes.
        """
        df = _apply_selection(df, self.features)
        return self._model.predict_proba(df.values)

    def score(self, df: pd.DataFrame, fn=None, **kwargs) -> float:
        """
        Predict and score using the ``f1_score()`` function. For multiclass
        problems, the ``average`` is "micro" by default unless specified
        otherwise by kwargs.

        :param df: A tabular dataset with features and target columns.
        :param fn: A custom scoring function.
        :param kwargs: Passed to the scoring function.
        :return: The resulting score.
        """
        y_pred = self.predict(df)
        y_true = np.squeeze(df[self.targets].values)
        try:
            if (
                len(self.targets) > 1
                or len(np.bincount(y_true)) > 2
                or len(np.bincount(y_pred)) > 2
                and "average" not in kwargs
            ):
                kwargs["average"] = "micro"
        except Exception as e:
            LOGGER.warning("Failed to infer the number of classes; Exception below")
            LOGGER.exception(e)
        if fn is not None:
            fn = fn
        elif self.score_fn is not None:
            fn = self.score_fn
        else:
            if "zero_division" not in kwargs:
                kwargs["zero_division"] = 0
            fn = f1_score
        return fn(y_true, y_pred, **kwargs)


class KinactiveRegressor(KinactiveModel):
    """
    A model wrapper for regression objective.
    """

    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        return _generate_fold_idx(df[self.cv_col].map(_get_unique_group).values, n)

    def score(self, df: pd.DataFrame, fn=None, **kwargs) -> float:
        """
        Predict and score on a dataset.

        :param df: A tabular dataset with features and target columns.
        :param fn: A custom scoring function. If not provided, RMSD will be used.
        :param kwargs: Passed to the scoring function.
        :return: The resulting score.
        """
        y_pred = self.predict(df)
        y_true = np.squeeze(df[self.targets].values)
        if fn:
            fn = curry(fn)(**kwargs)
        elif self.score_fn is not None:
            fn = curry(self.score_fn)(**kwargs)
        else:
            fn = lambda yt, tp: np.sqrt(np.mean((y_true - y_pred) ** 2))
        return fn(y_true, y_pred)


DFGModels = t.NamedTuple(
    "DFGModels",
    [
        ("in_", KinactiveClassifier),
        ("out", KinactiveClassifier),
        ("other", KinactiveClassifier),
        ("meta", KinactiveClassifier),
    ],
)


class DFGClassifier(ModelBase):
    """
    A composite model encapsulating three binary classifiers each predicting
    its own DFG conformation and a logistic regression meta-classifier trained
    on the [in, other, out] probabilities.

    Nevertheless, it behaves like a regular model providing interface similar to
    the :class:`KinActiveClassifier`.
    """

    def __init__(
        self,
        in_model: KinactiveClassifier,
        out_model: KinactiveClassifier,
        other_model: KinactiveClassifier,
        meta_model: KinactiveClassifier,
        cv_col: str = "ObjectID",
    ):
        self.models: DFGModels = DFGModels(in_model, out_model, other_model, meta_model)
        self.cv_col = cv_col

    @property
    def features(self) -> abc.Sequence[str]:
        """
        This returns :meth:`dfg_features` and exists for compatability with the
        :class:`ModelBase`.
        """
        return self.dfg_features

    @property
    def targets(
        self,
    ) -> list[str]:
        return self.models.meta.targets

    @property
    def dfg_features(self) -> list[str]:
        """
        :return: A list of features used by the XGBoost binary "in", "out", and
            "other" models.
        """
        return list(
            unique_everseen(chain.from_iterable(m.features for m in self.models[:3]))
        )

    @property
    def meta_features(self) -> list[str]:
        """
        :return: A list of features used by the "meta" LR classifier.
        """
        return self.models.meta.features

    @property
    def proba_names(self) -> list[str]:
        """
        :return: A list of column names of [in, out, other] probabilities.
        """
        return [ColNames.dfg_in_proba, ColNames.dfg_out_proba, ColNames.dfg_other_proba]

    def train(self, df: pd.DataFrame):
        """
        1. Train :attr:`models`
        2. Use trained :attr:`models` to predict their response variables.
        3. Use predicted variables to train the `meta` model.

        :param df: A dataset to train on. Must include all relevant variables.
        """
        for m in self.models[:3]:
            m.train(df)

        df = df.copy()
        for n, m in zip(self.proba_names, self.models[:3]):
            df[n] = m.predict_proba(df)[:, 1]

        self.models.meta.train(df)

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict all response variables.

        :param df: A dataset to predict on. Must include all relevant
            variables.
        :return: A copy of the ``df`` with predictions.
        """
        df = df.copy()
        for n, m in zip(self.proba_names, self.models[:3]):
            df[n] = m.predict_proba(df)[:, 1]

        y_prob = self.models.meta.predict_proba(df).round(2)
        for i, c in enumerate(ColNames.dfg_meta_proba_cols):
            df[c] = y_prob[:, i]

        df[ColNames.dfg_cls_pred] = np.argmax(y_prob, axis=1)
        df[ColNames.dfg_pred] = df[ColNames.dfg_cls_pred].map(DFG_MAP_REV)

        return df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict the DFG class. ``0`` stands for DFGin, ``1`` for DFGout, and
        ``2`` for DFGinter.

        .. note::
            This is equivalent to :meth:`predict_full` and selecting the
            relevant column.

        :param df: A dataset to predict from. Must include all relevant
            variables.
        :return: An array of predicted classes.
        """
        return self.predict_full(df)[ColNames.dfg_cls_pred].values

    def reinit_model(self):
        for m in self.models[:3]:
            m.reinit_model()

    def score(self, df: pd.DataFrame, **kwargs):
        y_true = df[ColNames.dfg_cls].values
        y_pred = self.predict(df)
        if "average" not in kwargs:
            kwargs["average"] = "micro"
        return f1_score(y_true, y_pred, **kwargs)

    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        return _generate_stratified_fold_idx(
            df[self.cv_col].values, np.squeeze(df[self.targets].values), n
        )

    def cv(self, df: pd.DataFrame, n: int, verbose: bool = True):
        return _cross_validate(self, df, n, verbose)

    def cv_pred(self, df: pd.DataFrame, n: int, verbose: bool = True):
        """
        Cross-validate the score and predict the data in test folds.

        :param df: Input data with features and target columns.
        :param n: The number of CV folds to use.
        :param verbose: Output progress bar.
        :return: A tuple with score and a copy of the supplied dataframe with
            fold assignment and model prediction columns added.
        """
        return _cross_validate_and_predict(self, df, n, verbose)


class EarlyStoppingCallback:
    """
    Early stopping callback for Optuna.

    See https://github.com/optuna/optuna/issues/1001#issuecomment-862843041
    """

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        if direction == "minimize":
            self._operator = op.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = op.gt
            self._score = -np.inf
        else:
            raise ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            LOGGER.info(
                f"Stopping optimization at max iter {self.early_stopping_rounds}"
            )
            study.stop()


class EvalSetSupplier:
    def __init__(
        self,
        obj_ids: np.ndarray | abc.Sequence[t.Hashable],
        target: abc.Sequence[int] | None,
        n_folds: int,
    ):
        self.target = target
        self.obj_ids = obj_ids
        self.n_folds = n_folds

    def generate_fold_idx(self):
        if self.target is None:
            return next(_generate_fold_idx(self.obj_ids, self.n_folds))
        return next(
            _generate_stratified_fold_idx(self.obj_ids, self.target, self.n_folds)
        )

    def __call__(
        self,
        estimator,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ):
        assert len(self.obj_ids) == len(trial_data.x_train)
        train_idx, eval_idx = self.generate_fold_idx()
        x_train, y_train = trial_data.x_train[train_idx], trial_data.y_train[train_idx]
        x_eval, y_eval = trial_data.x_train[eval_idx], trial_data.y_train[eval_idx]
        if trial_data.w_train is not None:
            w_train, w_eval = trial_data.w_train[train_idx], trial_data.w_test[eval_idx]
            kwargs["sample_weight_eval_set"] = [w_eval]
        else:
            w_train = None
        kwargs["eval_set"] = [(x_eval, y_eval)]
        kwargs["verbose"] = False
        new_trial_data = TrialData(x_train, x_train, y_train, y_train, w_train, w_train)
        return estimator, features, dataset, new_trial_data, kwargs


def _apply_selection(df: pd.DataFrame, features: list[str]):
    if not features:
        return df
    return df[features]


def _generate_fold_chunks(
    obj_ids: np.ndarray, n_folds: int
) -> abc.Generator[tuple[np.ndarray, np.ndarray], None, None]:
    ids = np.unique(obj_ids)
    np.random.shuffle(ids)
    chunks = np.array_split(ids, n_folds)
    for i in range(n_folds):
        chunk_test = chunks[i]
        chunk_train = np.concatenate([x for j, x in enumerate(chunks) if j != i])
        yield chunk_train, chunk_test


def _generate_fold_idx(
    obj_ids: np.ndarray, n_folds: int
) -> abc.Generator[tuple[np.ndarray, np.ndarray], None, None]:
    for train_chunk, test_chunk in _generate_fold_chunks(obj_ids, n_folds):
        idx_test = np.isin(obj_ids, test_chunk)
        idx_train = np.isin(obj_ids, train_chunk)
        yield idx_train, idx_test


def _generate_stratified_fold_idx(
    obj_ids: abc.Sequence[abc.Hashable],
    target: abc.Sequence[abc.Hashable],
    n_folds: int,
):
    df = pd.DataFrame({"ObjectID": obj_ids, "Target": target})
    groups = [
        _generate_fold_chunks(gg["ObjectID"], n_folds) for _, gg in df.groupby("Target")
    ]
    for id_pairs in map(list, zip(*groups)):
        ids_train = np.concatenate([x[0] for x in id_pairs])
        ids_test = np.concatenate([x[1] for x in id_pairs])
        idx_train = np.isin(obj_ids, ids_train)
        idx_test = np.isin(obj_ids, ids_test)

        yield idx_train, idx_test


def _parse_pdb_id(obj_id: str) -> str:
    finds = PDB_PATTERN.findall(obj_id)
    if not finds:
        raise ValueError(f"Failed to find any PDB IDs in {obj_id}")
    if len(finds) > 1:
        raise ValueError(f"Found multiple PDB IDs in the same object ID {obj_id}")
    return finds.pop()


def _get_unique_group(obj_id: str) -> str:
    try:
        return _parse_pdb_id(obj_id)
    except ValueError:
        return obj_id


def _get_xy(
    df,
    features: list[str],
    targets: list[str] | None,
    weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray | None]:
    ys = _apply_selection(df, targets) if targets else None
    ws = df[weight].values if weight else None
    df = _apply_selection(df, features)
    return df, ys, ws


def _cross_validate(
    model: ModelBase,
    df: pd.DataFrame,
    n: int,
    verbose: bool = False,
    return_scores: bool = False
) -> float | list[float]:
    idx_gen = model.generate_fold_idx(df, n)
    if verbose:
        idx_gen = tqdm(idx_gen, total=n, desc="Cross-validating")
    scores = []
    for train_idx, test_idx in idx_gen:
        model.reinit_model()
        model.train(df[train_idx])
        scores.append(model.score(df[test_idx]))
    score = float(np.mean(scores))
    msg = f"Scores: {np.array(scores).round(2)}; mean={score}"
    if verbose:
        LOGGER.info(msg)
    else:
        LOGGER.debug(msg)
    return scores if return_scores else score


def _cross_validate_and_predict(
    model: ModelBase,
    df: pd.DataFrame,
    n: int,
    verbose: bool = False,
    return_scores: bool = False,
) -> tuple[float | list[float], pd.DataFrame]:
    df = df.copy()
    idx_gen = model.generate_fold_idx(df, n)
    if verbose:
        idx_gen = tqdm(idx_gen, total=n, desc="Cross-validating")
    scores = []
    for fold_i, (train_idx, test_idx) in enumerate(idx_gen, start=1):
        model.reinit_model()
        model.train(df[train_idx])
        scores.append(model.score(df[test_idx]))

        if isinstance(model, DFGClassifier):
            df_pred = model.predict_full(df[test_idx])
            for col in model.proba_names:
                df.loc[test_idx, col] = df_pred[col]
            df.loc[test_idx, ColNames.dfg_cls_pred] = df_pred[ColNames.dfg_cls_pred]
        else:
            df.loc[test_idx, f"{model.targets[0]}_pred"] = model.predict(df[test_idx])

        df.loc[test_idx, "Fold_i"] = fold_i

    score = mean(scores)
    msg = f"Scores: {scores}; mean={score}"
    if verbose:
        LOGGER.info(msg)
    else:
        LOGGER.debug(msg)
    if return_scores:
        return scores, df
    return score, df


def make(
    df: pd.DataFrame,
    targets: list[str],
    features: list[str],
    starting_params: dict[str, t.Any],
    cv_col: str = "ObjectID",
    weight_col: str | None = None,
    use_early_stopping: bool = False,
    early_stopping_rounds_param_sel: int = 0,
    base_model=None,
    classifier: bool = True,
    n_trials_sel_1: int = 50,
    n_trials_sel_2: int = 50,
    n_cv_sel_1: int = 10,
    n_cv_sel_2: int = 10,
    n_final_cv: int = 10,
    n_folds_fs: int = 10,
    boruta_kwargs: dict[str, t.Any] | None = None,
) -> tuple[KinactiveClassifier | KinactiveRegressor, float, pd.DataFrame, pd.DataFrame]:
    """
    A pipeline to make a new ``KinActive`` model. It comprises:

        #. Initializing the model using starting params.
        #. A parameter-selection run.
        #. A feature selection run.
        #. Another parameter selection run.
        #. Cross-validate and predict on test folds.
        #. Train on the full dataset.

    :param df: A table to train on.
    :param targets: The names of the target columns.
    :param features: The names of the feature columns.
    :param starting_params: The starting model's parameters.
    :param cv_col: A column used to generate CV folds. The folds will be built
        such that the values this column points to will never overlap between
        folds.
    :param weight_col: Optionally, a column name pointing to the sample weights.
    :param use_early_stopping: Use early stopping to cap the number of trees.
        The ``early_stopping_rounds`` param may be provided via
        ``starting_params``. Valid for the XGBoost models.
    :param early_stopping_rounds_param_sel: The number of early stopping rounds
        for the hyperparameter optimization. ``0`` indicates no early stopping.
    :param base_model: Initialized based model. Anything supported by
        :class:`KinactiveClassifier` or :class:`KinactiveRegressor`.
    :param classifier: If ``True``, assume classification objective and init
        the :class:`KinactiveClassifier`. Otherwise, assume the regression and
        init the :class:`KinactiveRegressor`.
    :param n_trials_sel_1: The number of parameter selection rounds before the
        feature selection.
    :param n_trials_sel_2: The number of parameter selection rounds after the
        feature selection.
    :param n_cv_sel_1: The number of CV folds used to evaluate the performance
        after the first round of parameter selection.
    :param n_cv_sel_2: The number of CV folds used to evaluate the performance
        after the second round of parameter selection.
    :param n_folds_fs: The number of CV folds used to evaluate the performance
        during feature selection of an XGBoost model if early stopping is used.
    :param n_final_cv: The number of CV folds for the final CV.
    :param boruta_kwargs: Passed to the ``eBoruta`` feature selector.
    :return:
    """

    taken_names = ["ObjectID", *targets]
    features = features or [c for c in df.columns if c not in taken_names]

    args = dict(
        targets=targets,
        features=features,
        cv_col=cv_col,
        weight_col=weight_col,
        params=starting_params,
        use_early_stopping=use_early_stopping,
    )
    if classifier:
        model = KinactiveClassifier(
            XGBClassifier() if base_model is None else base_model, **args
        )
    else:
        model = KinactiveRegressor(
            XGBRegressor() if base_model is None else base_model, **args
        )

    if n_trials_sel_1 > 0:
        LOGGER.info(
            f"Selecting params using full feature set for {n_trials_sel_1} trials"
        )
        model.select_params(
            df,
            n_trials_sel_1,
            early_stopping_rounds=early_stopping_rounds_param_sel,
            n_cv=n_cv_sel_1,
        )
        LOGGER.info(f"Final params: {model.params}")

    LOGGER.info("Selecting features")
    kwargs = boruta_kwargs or {}
    model.select_features(df, n_folds_fs, **kwargs)
    LOGGER.info(f"Selected {len(model.features)} features")

    if n_trials_sel_2 > 0:
        LOGGER.info("Selecting params 2")
        model.select_params(
            df,
            n_trials_sel_2,
            early_stopping_rounds=early_stopping_rounds_param_sel,
            n_cv=n_cv_sel_2,
        )
        LOGGER.info(f"Final params: {model.params}")

    cv_score, df_pred = model.cv_pred(df, n_final_cv, verbose=True)
    LOGGER.info(f"Final CV score: {cv_score}")
    LOGGER.info("Fitting the final model")
    model.train(df)

    LOGGER.info("Ranking selected features")
    ranks = model.selector.rank(model.features, model=model.model, sort=True)

    return model, cv_score, df_pred, ranks


if __name__ == "__main__":
    raise RuntimeError
