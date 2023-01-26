from __future__ import annotations

import json
import logging
import re
import typing as t
from abc import abstractmethod
from collections import abc
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import pandas as pd
from Boruta import Boruta
from lXtractor.core.exceptions import MissingData
from lXtractor.util.io import get_files
from sklearn.metrics import f1_score, r2_score
from toolz import curry
from xgboost import XGBClassifier, XGBRegressor

from kinactive.config import _DumpNames

_PDB_PATTERN = re.compile(r'\((\w{4}):\w+\|')
_X = t.TypeVar('_X', XGBRegressor, XGBClassifier)
LOGGER = logging.getLogger(__name__)
DumpNames = _DumpNames()


# TODO Boruta: y as df is not supported
# TODO Boruta: explicit support for regressors (with use_test=True)


def _apply_selection(df: pd.DataFrame, features: list[str]):
    if not features:
        return df
    return df[features]


def _generate_fold_idx(
    obj_ids: np.ndarray, n_folds: int
) -> tuple[np.ndarray, np.ndarray]:
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

    def cv(self, df: pd.DataFrame, n: int, verbose: bool = False) -> float:
        idx_gen = _generate_fold_idx(df['ObjectID'].map(_get_unique_group).values, n)
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
    def predict_proba(self, df: pd.DataFrame):
        df = _apply_selection(df, self.features)
        return self._model.predict_proba(df.values)

    def score(self, df: pd.DataFrame, **kwargs) -> float:
        y_pred = self.predict(df)
        y_true = np.squeeze(df[self.targets].values)
        if (
            len(self.targets) > 1
            or len(np.bincount(y_true)) > 2
            and 'average' not in kwargs
        ):
            kwargs['average'] = 'micro'
        return f1_score(y_true, y_pred, **kwargs)


class KinactiveRegressor(KinactiveModel):
    def score(self, df: pd.DataFrame, **kwargs) -> float:
        y_pred = self.predict(df)
        y_true = np.squeeze(df[self.targets].values)
        return r2_score(y_true, y_pred, **kwargs)


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
) -> tuple[KinactiveClassifier | KinactiveRegressor, float]:
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

    cv_score = model.cv(df, n_final_cv)
    LOGGER.info(f'Final CV score: {cv_score}')
    model.train(df)

    return model, cv_score


def save_txt_lines(lines: abc.Iterable[str], path: Path) -> Path:
    with path.open('w') as f:
        for x in lines:
            print(x, file=f)
    return path


def save_json(data: dict, path: Path) -> Path:
    with path.open('w') as f:
        json.dump(data, f)
    return path


def save_xgb(model: XGBClassifier | XGBRegressor, path: Path) -> Path:
    model.save_model(path)
    return path


def save(
    kin_model: KinactiveClassifier | KinactiveRegressor,
    base: Path,
    name: str,
    overwrite: bool = False,
) -> Path:
    if isinstance(kin_model, KinactiveRegressor):
        suffix = 'regressor'
    elif isinstance(kin_model, KinactiveClassifier):
        suffix = 'classifier'
    else:
        raise TypeError(f'Unexpected model type {kin_model.__class__}')
    path = base / f'{name}_{suffix}'
    if path.exists() and not overwrite:
        raise ValueError(
            f'Model path {path} exists. Set overwrite=True if you want to '
            'overwrite existing model'
        )
    path.mkdir(exist_ok=True, parents=True)

    save_txt_lines(kin_model.targets, path / DumpNames.targets_filename)
    save_txt_lines(kin_model.features, path / DumpNames.features_filename)
    save_json(kin_model.params, path / DumpNames.params_filename)
    save_xgb(kin_model.model, path / DumpNames.model_filename)

    return path


def load_txt_lines(path: Path) -> list[str]:
    return list(filter(bool, path.read_text().split('\n')))


def load_json(path: Path) -> dict[str, t.Any]:
    with path.open() as f:
        return json.load(f)


def load_xgb(path: Path, xgb_model: _X) -> _X:
    xgb_model.load_model(path)
    return xgb_model


def load(path: Path):
    if not path.is_dir():
        raise NotADirectoryError(f'{path} must be dir')
    name = path.name.lower()
    if 'classifier' in name:
        cls = KinactiveClassifier
        xgb_type = XGBClassifier
    elif 'regressor' in name:
        cls = KinactiveRegressor
        xgb_type = XGBRegressor
    else:
        raise NameError(
            'Directory name must contain either "regressor" or "classifier"'
        )
    files = get_files(path)
    expected_names = [
        DumpNames.model_filename,
        DumpNames.features_filename,
        DumpNames.params_filename,
    ]
    for name in expected_names:
        if name not in files:
            raise MissingData(f'Missing required file "{name}" in {path}')

    targets = load_txt_lines(files[DumpNames.targets_filename])
    features = load_txt_lines(files[DumpNames.features_filename])
    params = load_json(files[DumpNames.params_filename])
    xgb_model = load_xgb(files[DumpNames.model_filename], xgb_type())

    return cls(xgb_model, targets, features, params)


if __name__ == '__main__':
    raise RuntimeError
