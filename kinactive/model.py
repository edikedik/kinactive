from __future__ import annotations

import logging
import re
import typing as t
from abc import abstractmethod
from collections import abc
from statistics import mean

import numpy as np
import optuna
import pandas as pd
from Boruta import Boruta
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from toolz import curry
from xgboost import XGBClassifier, XGBRegressor

_PDB_PATTERN = re.compile(r'\((\w{4}):\w+\|')
LOGGER = logging.getLogger(__name__)


# TODO Boruta: y as df is not supported
# TODO Boruta: explicit support for regressors (with use_test=True)


def _apply_selection(df: pd.DataFrame, features: list[str]):
    if not features:
        return df
    return df[features]


def _generate_fold_idx(
    obj_ids: np.ndarray, n_folds: int
) -> abc.Generator[tuple[np.ndarray, np.ndarray], None, None]:
    ids_original = obj_ids.copy()
    ids = np.unique(obj_ids)
    np.random.shuffle(ids)
    chunks = np.array_split(ids, n_folds)
    for i in range(n_folds):
        chunk_test = chunks[i]
        chunk_train = np.concatenate([x for j, x in enumerate(chunks) if j != i])
        idx_test = np.isin(ids_original, chunk_test)
        idx_train = np.isin(ids_original, chunk_train)
        yield idx_train, idx_test


def _generate_stratified_fold_idx(
    obj_ids: np.ndarray | abc.Sequence[str],
    target: np.ndarray | abc.Sequence[str],
    n_folds: int,
):
    def concat_ith(a: abc.Sequence[np.ndarray], i: int) -> np.ndarray:
        return np.concatenate([c[i] for c in a])

    df = pd.DataFrame({'ObjectID': obj_ids, 'Target': target})
    df_unique = df.drop_duplicates().sample(frac=1)
    class_chunks = []
    for _, gg in df_unique.groupby(target):
        uniq_objects = gg['ObjectID'].unique()
        class_chunks.append(np.array_split(uniq_objects, n_folds))
    print('\n', *class_chunks, sep='\n', end='\n\n')
    for i in range(n_folds):
        chunk_test = concat_ith(class_chunks, i)
        chunk_train = np.concatenate(
            [concat_ith(class_chunks, j) for j in range(n_folds) if i != j]
        )
        chunk_test = np.setdiff1d(chunk_test, chunk_train)
        idx_test = df['ObjectID'].isin(chunk_test)
        idx_train = df['ObjectID'].isin(chunk_train)
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


@curry
def xgb_objective(
    trial: optuna.Trial, df: pd.DataFrame, model: KinactiveModel
) -> float:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'gamma': trial.suggest_float('gamma', 0, 20.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.0, 10.0),
    }

    # callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_logloss')
    params = {**model.params, **params}
    model = model.__class__(model.model, model.targets, model.features, params)
    score = model.cv(df, 5)
    return score


class KinactiveModel:
    def __init__(
        self,
        model,
        targets: abc.Iterable[str],
        features: abc.Iterable[str] = (),
        params: dict[str, t.Any] | None = None,
    ):
        if not isinstance(features, list):
            features = list(features)
        if not isinstance(targets, list):
            targets = list(targets)
        self._features = features
        self._targets = targets
        self._model = model
        self.params = params or {}

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
        df, ys = _get_xy(df, self.features, self.targets)
        assert ys is not None, f'failed finding target variables {self.targets}'
        self._model.fit(df.values, np.squeeze(ys.values))

    def predict(self, df: pd.DataFrame):
        df = _apply_selection(df, self.features)
        return self._model.predict(df.values)

    @abstractmethod
    def score(self, df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def generate_fold_idx(
        self, df: pd.DataFrame, n: int
    ) -> abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def cv(self, df: pd.DataFrame, n: int, verbose: bool = False) -> float:
        idx_gen = self.generate_fold_idx(df, n)
        scores = []
        for train_idx, test_idx in idx_gen:
            self.reinit_model()
            self.train(df[train_idx])
            scores.append(self.score(df[test_idx]))
        score = mean(scores)
        msg = f'Scores: {scores}; mean={score}'
        if verbose:
            LOGGER.info(msg)
        else:
            LOGGER.debug(msg)
        return score

    def cv_pred(
        self, df: pd.DataFrame, n: int, verbose: bool = False
    ) -> tuple[float, pd.DataFrame]:
        df = df.copy()
        idx_gen = self.generate_fold_idx(df, n)
        scores = []
        for fold_i, (train_idx, test_idx) in enumerate(idx_gen, start=1):
            self.reinit_model()
            self.train(df[train_idx])
            scores.append(self.score(df[test_idx]))
            y_pred = self.predict(df[test_idx])
            if len(self.targets) == 1:
                df.loc[test_idx, f'{self.targets[0]}_pred'] = y_pred
            else:
                for i, col in enumerate(self.targets):
                    df.loc[test_idx, f'{col}_pred'] = y_pred[:, i]
            df.loc[test_idx, 'Fold_i'] = fold_i
        score = mean(scores)
        msg = f'Scores: {scores}; mean={score}'
        if verbose:
            LOGGER.info(msg)
        else:
            LOGGER.debug(msg)
        return score, df

    def select_params(
        self, df: pd.DataFrame, n_trials: int, direction: str = 'maximize'
    ) -> optuna.Study:
        objective = xgb_objective(df=df, model=self)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
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
        splitter = StratifiedShuffleSplit(n_splits=n, test_size=1 / n)
        x, y = _get_xy(df, self.features, self.targets)
        for train_idx, test_idx in splitter.split(x, y):
            yield df.index.isin(train_idx), df.index.isin(test_idx)

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


class DFGClassifier:
    def __init__(
        self,
        in_model: KinactiveClassifier,
        out_model: KinactiveClassifier,
        inter_model: KinactiveClassifier,
        meta_model: KinactiveClassifier,
    ):
        self.models = DFGModels(in_model, out_model, inter_model, meta_model)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['in_proba'] = self.models.in_.predict_proba(df)
        df['out_proba'] = self.models.out.predict_proba(df)
        df['inter_proba'] = self.models.inter.predict_proba(df)
        y_prob = self.models.meta.predict_proba(df).round(2)
        for i, label in enumerate(['in', 'out', 'inter']):
            df[f'{label}_proba_meta'] = y_prob[:, i]
        df['DFG_pred'] = np.argmax(y_prob, axis=1)
        return df


def make(
    df: pd.DataFrame,
    targets: list[str],
    features: list[str],
    starting_params: dict[str, t.Any],
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
        )
    else:
        model = KinactiveRegressor(
            XGBRegressor(),
            targets,
            features,
            params=starting_params,
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
