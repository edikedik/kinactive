from __future__ import annotations

import json
import typing as t
from collections import abc
from pathlib import Path

from lXtractor.core.exceptions import MissingData
from lXtractor.util.io import get_files
from xgboost import XGBClassifier, XGBRegressor

from kinactive.config import _DumpNames
from kinactive.model import KinactiveClassifier, KinactiveRegressor, DFGClassifier

DumpNames = _DumpNames()
_X = t.TypeVar('_X', XGBRegressor, XGBClassifier)


def save_txt_lines(lines: abc.Iterable[t.Any], path: Path) -> Path:
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


def save_dfg(
    dfg_model: DFGClassifier, base: Path, name: str, overwrite: bool = False
) -> Path:
    save(dfg_model.models.in_, base / name, DumpNames.in_model_dirname, overwrite)
    save(dfg_model.models.out, base / name, DumpNames.out_model_dirname, overwrite)
    save(dfg_model.models.inter, base / name, DumpNames.inter_model_dirname, overwrite)
    save(dfg_model.models.d1, base / name, DumpNames.d1_model_dirname, overwrite)
    save(dfg_model.models.d2, base / name, DumpNames.d2_model_dirname, overwrite)
    save(dfg_model.models.meta, base / name, DumpNames.meta_model_dirname, overwrite)
    return base / name


def save(
    model: KinactiveClassifier | KinactiveRegressor | DFGClassifier,
    base: Path,
    name: str,
    overwrite: bool = False,
) -> Path:
    if isinstance(model, KinactiveRegressor):
        suffix = 'regressor'
    elif isinstance(model, KinactiveClassifier):
        suffix = 'classifier'
    elif isinstance(model, DFGClassifier):
        return save_dfg(model, base, name, overwrite)
    else:
        raise TypeError(f'Unexpected model type {model.__class__}')
    path = base / f'{name}_{suffix}'
    if path.exists() and not overwrite:
        raise ValueError(
            f'Model path {path} exists. Set overwrite=True if you want to '
            'overwrite existing model'
        )
    path.mkdir(exist_ok=True, parents=True)

    save_txt_lines(model.targets, path / DumpNames.targets_filename)
    save_txt_lines(model.features, path / DumpNames.features_filename)
    save_json(model.params, path / DumpNames.params_filename)
    save_xgb(model.model, path / DumpNames.model_filename)

    return path


def load_txt_lines(path: Path) -> list[str]:
    return list(filter(bool, path.read_text().split('\n')))


def load_json(path: Path) -> dict[str, t.Any]:
    with path.open() as f:
        return json.load(f)


def load_xgb(path: Path, xgb_model: _X) -> _X:
    xgb_model.load_model(path)
    return xgb_model


def load_dfg(path: Path) -> DFGClassifier:
    models = (
        load(path / name)
        for name in [
            DumpNames.in_model_dirname,
            DumpNames.out_model_dirname,
            DumpNames.inter_model_dirname,
            DumpNames.d1_model_dirname,
            DumpNames.d2_model_dirname,
            DumpNames.meta_model_dirname,
        ]
    )
    in_, out, inter, d1, d2, meta = models
    return DFGClassifier(in_, out, inter, d1, d2, meta)


def load(path: Path) -> KinactiveClassifier | KinactiveRegressor | DFGClassifier:
    if not path.is_dir():
        raise NotADirectoryError(f'{path} must be dir')
    name = path.name.lower()

    if 'dfg' in name:
        return load_dfg(path)

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
