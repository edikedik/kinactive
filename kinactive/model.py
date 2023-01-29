from __future__ import annotations

import logging
import operator as op
import re
import typing as t
from abc import ABCMeta, abstractmethod
from collections import abc
from statistics import mean

import numpy as np
import optuna
import pandas as pd
from Boruta import Boruta
from sklearn.metrics import f1_score, r2_score
from toolz import curry
from xgboost import XGBClassifier, XGBRegressor

from kinactive.config import ColNames, DFG_MAP_REV

_PDB_PATTERN = re.compile(r'\((\w{4}):\w+\|')
LOGGER = logging.getLogger(__name__)


# TODO Boruta: y as df is not supported
# TODO Boruta: explicit support for regressors (with use_test=True)


class ModelBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def targets(self) -> abc.Sequence[str]:
        ...

    @abstractmethod
    def reinit_model(self):
        ...

    @abstractmethod
    def train(self, df: pd.DataFrame):
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        ...

    @abstractmethod
    def cv(self, df: pd.DataFrame, n: int):
        ...

    @abstractmethod
    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        ...


class KinactiveModel(ModelBase, metaclass=ABCMeta):
    def __init__(
        self,
        model,
        targets: abc.Iterable[str],
        features: abc.Iterable[str] = (),
        params: dict[str, t.Any] | None = None,
        use_early_stopping: bool = False,
    ):
        if not isinstance(features, list):
            features = list(features)
        if not isinstance(targets, list):
            targets = list(targets)
        self._features = features
        self._targets = targets
        self._model = model
        self.params = params or {}
        self.use_early_stopping = use_early_stopping

    @property
    def model(self):
        return self._model

    @property
    def features(self) -> list[str]:
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
        if self.use_early_stopping:
            train_idx, eval_idx = next(self.generate_fold_idx(df, 10))
            train_df, train_ys = _get_xy(df[train_idx], self.features, self.targets)
            eval_df, eval_ys = _get_xy(df[eval_idx], self.features, self.targets)
            self._model.fit(
                train_df,
                np.squeeze(train_ys.values),
                eval_set=[(eval_df, np.squeeze(eval_ys.values))],
                verbose=False,
            )
        else:
            assert (
                'early_stopping_rounds' not in self.params
            ), 'Must not have early stopping params if `use_early_stopping` is `False`'
            xs, ys = _get_xy(df, self.features, self.targets)
            assert ys is not None, f'failed finding target variables {self.targets}'
            self._model.fit(xs.values, np.squeeze(ys.values))

    def predict(self, df: pd.DataFrame):
        df = _apply_selection(df, self.features)
        return self._model.predict(df.values)

    def cv(self, df: pd.DataFrame, n: int, verbose: bool = False) -> float:
        return _cross_validate(self, df, n, verbose)

    def cv_pred(
        self, df: pd.DataFrame, n: int, verbose: bool = False
    ) -> tuple[float, pd.DataFrame]:
        return _cross_validate_and_predict(self, df, n, verbose)

    def select_params(
        self, df: pd.DataFrame, n_trials: int, direction: str = 'maximize'
    ) -> optuna.Study:
        objective = xgb_objective(
            df=df, model=self, use_early_stopping=self.use_early_stopping
        )
        study = optuna.create_study(direction=direction)
        cb = EarlyStoppingCallback(10, direction=direction)
        study.optimize(objective, n_trials=n_trials, callbacks=[cb])
        self.params = study.best_params
        return study

    def select_features(self, df: pd.DataFrame, **kwargs) -> Boruta:
        boruta = Boruta(**kwargs)
        df_x, df_y = _get_xy(df, self.features, self.targets)
        assert df_y is not None
        res = boruta.fit(df_x, np.squeeze(df_y.values), model=self.model)
        self._features = list(res.features_.accepted)
        return res


class KinactiveClassifier(KinactiveModel):
    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        return _generate_stratified_fold_idx(
            df['ObjectID'].values, np.squeeze(df[self.targets].values), n
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = _apply_selection(df, self.features)
        return self._model.predict_proba(df.values)

    def score(self, df: pd.DataFrame, **kwargs) -> float:
        y_pred = self.predict(df)
        y_true = np.squeeze(df[self.targets].values)
        if (
            len(self.targets) > 1
            or len(np.bincount(y_true)) > 2
            or len(np.bincount(y_pred)) > 2
            and 'average' not in kwargs
        ):
            kwargs['average'] = 'micro'
        return f1_score(y_true, y_pred, **kwargs)


class KinactiveRegressor(KinactiveModel):
    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        return _generate_fold_idx(df['ObjectID'].map(_get_unique_group).values, n)

    def score(self, df: pd.DataFrame, **kwargs) -> float:
        y_pred = self.predict(df)
        y_true = np.squeeze(df[self.targets].values)
        return r2_score(y_true, y_pred, **kwargs)


DFGModels = t.NamedTuple(
    'DFGModels',
    [
        ('in_', KinactiveClassifier),
        ('out', KinactiveClassifier),
        ('inter', KinactiveClassifier),
        ('meta', KinactiveClassifier),
    ],
)


class DFGClassifier(ModelBase):
    def __init__(
        self,
        in_model: KinactiveClassifier,
        out_model: KinactiveClassifier,
        inter_model: KinactiveClassifier,
        meta_model: KinactiveClassifier,
    ):
        self.models = DFGModels(in_model, out_model, inter_model, meta_model)

    @property
    def targets(self) -> list[str]:
        return [ColNames.dfg_cls]

    def reinit_model(self):
        for m in self.models:
            m.reinit_model()

    def _predict_no_meta(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.models.in_.predict_proba(df)[:, 1],
            self.models.out.predict_proba(df)[:, 1],
            self.models.inter.predict_proba(df)[:, 1],
        )

    def train(self, df: pd.DataFrame):
        self.models.in_.train(df)
        self.models.out.train(df)
        self.models.inter.train(df)
        df_pred = df.copy()
        pred = self._predict_no_meta(df)
        for c, p in zip(ColNames.dfg_proba_cols, pred):
            df_pred[c] = p
        self.models.meta.train(df)

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pred = self._predict_no_meta(df)
        for c, p in zip(ColNames.dfg_proba_cols, pred):
            df[c] = p
        y_prob = self.models.meta.predict_proba(df).round(2)
        for i, c in enumerate(ColNames.dfg_meta_proba_cols):
            df[c] = y_prob[:, i]
        df[ColNames.dfg_cls_pred] = np.argmax(y_prob, axis=1)
        df[ColNames.dfg_pred] = df[ColNames.dfg_cls_pred].map(DFG_MAP_REV)
        return df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_full(df)[ColNames.dfg_cls_pred].values

    def score(self, df: pd.DataFrame, **kwargs):
        y_true = df[ColNames.dfg_cls].values
        y_pred = self.predict(df)
        if 'average' not in kwargs:
            kwargs['average'] = 'micro'
        return f1_score(y_true, y_pred, **kwargs)

    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        return _generate_stratified_fold_idx(
            df['ObjectID'].values, np.squeeze(df[self.targets].values), n
        )

    def cv(self, df: pd.DataFrame, n: int, verbose: bool = True):
        return _cross_validate(self, df, n, verbose)

    def cv_pred(self, df: pd.DataFrame, n: int, verbose: bool = True):
        return _cross_validate_and_predict(self, df, n, verbose)


class EarlyStoppingCallback(object):
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
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


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
    df = pd.DataFrame({'ObjectID': obj_ids, 'Target': target})
    groups = [
        _generate_fold_chunks(gg['ObjectID'], n_folds) for _, gg in df.groupby('Target')
    ]
    for id_pairs in map(list, zip(*groups)):
        ids_train = np.concatenate([x[0] for x in id_pairs])
        ids_test = np.concatenate([x[1] for x in id_pairs])
        idx_train = np.isin(obj_ids, ids_train)
        idx_test = np.isin(obj_ids, ids_test)

        yield idx_train, idx_test


def _parse_pdb_id(obj_id: str) -> str:
    finds = _PDB_PATTERN.findall(obj_id)
    if not finds:
        raise ValueError(f'Failed to find any PDB IDs in {obj_id}')
    if len(finds) > 1:
        raise ValueError(f'Found multiple PDB IDs in the same object ID {obj_id}')
    return finds.pop()


def _get_unique_group(obj_id: str) -> str:
    try:
        return _parse_pdb_id(obj_id)
    except ValueError:
        return obj_id


def _get_xy(
    df, features: list[str], targets: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    ys = _apply_selection(df, targets) if targets else None
    df = _apply_selection(df, features)
    return df, ys


def _cross_validate(
    model: KinactiveModel | DFGClassifier,
    df: pd.DataFrame,
    n: int,
    verbose: bool = False,
) -> float:
    idx_gen = model.generate_fold_idx(df, n)
    scores = []
    for train_idx, test_idx in idx_gen:
        model.reinit_model()
        model.train(df[train_idx])
        scores.append(model.score(df[test_idx]))
    score = np.mean(scores)
    msg = f'Scores: {np.array(scores).round(2)}; mean={score}'
    if verbose:
        LOGGER.info(msg)
    else:
        LOGGER.debug(msg)
    return score


def _cross_validate_and_predict(
    model: KinactiveModel | DFGClassifier,
    df: pd.DataFrame,
    n: int,
    verbose: bool = False,
) -> tuple[float, pd.DataFrame]:
    df = df.copy()
    idx_gen = model.generate_fold_idx(df, n)
    scores = []
    for fold_i, (train_idx, test_idx) in enumerate(idx_gen, start=1):
        model.reinit_model()
        model.train(df[train_idx])
        scores.append(model.score(df[test_idx]))
        y_pred = model.predict(df[test_idx])
        if len(model.targets) == 1:
            df.loc[test_idx, f'{model.targets[0]}_pred'] = y_pred
        else:
            for i, col in enumerate(model.targets):
                df.loc[test_idx, f'{col}_pred'] = y_pred[:, i]
        df.loc[test_idx, 'Fold_i'] = fold_i
    score = mean(scores)
    msg = f'Scores: {scores}; mean={score}'
    if verbose:
        LOGGER.info(msg)
    else:
        LOGGER.debug(msg)
    return score, df


@curry
def xgb_objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    model: KinactiveModel,
    n_cv: int = 5,
    use_early_stopping: bool = False,
) -> float:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0, 1),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'gamma': trial.suggest_float('gamma', 0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
    }
    if isinstance(model, KinactiveClassifier):
        params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.0, 10.0)
    if not use_early_stopping:
        params['n_estimators'] = trial.suggest_int('max_depth', 10, 1000)

    # callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_logloss')
    params = {**model.params, **params}
    model = model.__class__(
        model.model, model.targets, model.features, params, use_early_stopping
    )
    score = model.cv(df, n_cv)
    return score


def make(
    df: pd.DataFrame,
    targets: list[str],
    features: list[str],
    starting_params: dict[str, t.Any],
    use_early_stopping: bool = False,
    classifier: bool = True,
    n_trials_sel_1: int = 50,
    n_trials_sel_2: int = 50,
    n_final_cv: int = 10,
    boruta_kwargs: dict[str, t.Any] | None = None,
) -> tuple[KinactiveClassifier | KinactiveRegressor, float, pd.DataFrame]:
    taken_names = ['ObjectID', *targets]
    features = features or [c for c in df.columns if c not in taken_names]

    if classifier:
        model = KinactiveClassifier(
            XGBClassifier(),
            targets,
            features,
            params=starting_params,
            use_early_stopping=use_early_stopping,
        )
    else:
        model = KinactiveRegressor(
            XGBRegressor(),
            targets,
            features,
            params=starting_params,
            use_early_stopping=use_early_stopping,
        )

    if n_trials_sel_1 > 0:
        LOGGER.info(
            f'Selecting params using full feature set for {n_trials_sel_1} trials'
        )
        model.select_params(df, n_trials_sel_1)
        LOGGER.info(f'Final params: {model.params}')

    LOGGER.info('Selecting features')
    kwargs = boruta_kwargs or {}
    model.select_features(df, **kwargs)
    LOGGER.info(f'Selected {len(model.features)} features')

    if n_trials_sel_2 > 0:
        LOGGER.info('Selecting params 2')
        model.select_params(df, n_trials_sel_2)
        LOGGER.info(f'Final params: {model.params}')

    cv_score, df_pred = model.cv_pred(df, n_final_cv, verbose=True)
    LOGGER.info(f'Final CV score: {cv_score}')
    model.train(df)

    return model, cv_score, df_pred


if __name__ == '__main__':
    raise RuntimeError
