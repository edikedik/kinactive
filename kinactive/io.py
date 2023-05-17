"""
Files and models' IO operations.
"""
from __future__ import annotations

import json
import typing as t
from collections import abc
from pathlib import Path

import joblib
from lXtractor.core.exceptions import MissingData
from lXtractor.util.io import get_files
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from kinactive.config import DumpNames, ModelPaths
from kinactive.model import DFGClassifier, KinactiveClassifier, KinactiveRegressor

_X = t.TypeVar("_X", XGBRegressor, XGBClassifier)


def save_txt_lines(lines: abc.Iterable[t.Any], path: Path) -> Path:
    """
    :param lines: Iterable over any printable elements.
    :param path: A valid path to write to.
    :return: The ``path`` after successful writing.
    """
    with path.open("w") as f:
        for x in lines:
            print(x, file=f)
    return path


def save_json(data: dict, path: Path) -> Path:
    """
    :param data: Data dictionary.
    :param path: A valid path to write to.
    :return: The ``path`` after successful writing.
    """
    with path.open("w") as f:
        json.dump(data, f)
    return path


def save_xgb(model: XGBClassifier | XGBRegressor, path: Path) -> Path:
    """
    :param model: XGBoost model with a ``save_model()`` method.
    :param path: A valid path to write to.
    :return: The ``path`` after successful writing.
    """
    model.save_model(path)
    return path


def save_sklearn(model: LogisticRegression, path: Path) -> Path:
    """
    Save an sklearn model using ``joblib.dump()``.

    :param model: A scikit-learn model.
    :param path: A valid path to write to.
    :return: The ``path`` after successful writing.
    """
    return Path(joblib.dump(model, path)[0])


def save_dfg(
    model: DFGClassifier,
    base: Path,
    name: str,
    overwrite: bool = False,
) -> Path:
    """
    Save the DFGclassifier model into four different folders.

    This will save each model separately using :func:`save` under paths
    ``base / name / model_name``.

    :param model: The model to save.
    :param base: Base path to write to.
    :param name: A dir name within the ``base`` dir.
    :param overwrite: Overwrite existing models.
    :return: ``base/name`` path after successful save.
    """
    save(model.models.in_, base / name, DumpNames.in_model_dirname, overwrite)
    save(model.models.out, base / name, DumpNames.out_model_dirname, overwrite)
    save(model.models.other, base / name, DumpNames.other_model_dirname, overwrite)
    save(model.models.meta, base / name, DumpNames.meta_model_dirname, overwrite)

    return base / name


def save(
    model: KinactiveClassifier
    | KinactiveRegressor
    | LogisticRegression
    | DFGClassifier,
    base: Path,
    name: str,
    overwrite: bool = False,
) -> Path:
    """
    :param model: A model from :mod:`kinactive.model`.
    :param base: Base dir to save to.
    :param name: Model name. Will create ``base / name`` dir if it doesn't exist.
    :param overwrite: Overwrite existing model with the same name.
    :return: The ``base / name`` path after successful save.
    """
    if isinstance(model, KinactiveRegressor):
        suffix = "regressor"
    elif isinstance(model, (KinactiveClassifier, LogisticRegression)):
        suffix = "classifier"
    elif isinstance(model, DFGClassifier):
        return save_dfg(model, base, name, overwrite)
    else:
        raise TypeError(f"Unexpected model type {model.__class__}")
    path = base / f"{name}_{suffix}"
    if path.exists() and not overwrite:
        raise ValueError(
            f"Model path {path} exists. Set overwrite=True if you want to "
            "overwrite existing model"
        )
    path.mkdir(exist_ok=True, parents=True)

    save_txt_lines(model.targets, path / DumpNames.targets_filename)
    save_txt_lines(model.features, path / DumpNames.features_filename)
    save_json(model.params, path / DumpNames.params_filename)

    if isinstance(model.model, LogisticRegression):
        save_sklearn(model.model, path / DumpNames.model_filename)
    elif isinstance(model.model, (XGBClassifier, XGBRegressor)):
        save_xgb(model.model, path / DumpNames.model_filename)
    else:
        raise TypeError("...")

    return path


def load_txt_lines(path: Path) -> list[str]:
    """
    :param path: A path to a text file.
    :return: A list of non-empty lines.
    """
    return list(filter(bool, path.read_text().split("\n")))


def load_json(path: Path) -> dict[str, t.Any]:
    """
    :param path: A path to a JSON file.
    :return: The parsed dictionary.
    """
    with path.open() as f:
        return json.load(f)


def load_xgb(path: Path, xgb_model: _X) -> _X:
    """
    :param path: A path to an XGBoost model saved via :func:`save_xgb`
    :param xgb_model: The model type.
    :return: The loaded model.
    """
    xgb_model.load_model(path)
    return xgb_model


def load_sklearn(path: Path) -> LogisticRegression:
    """
    :param path: A path to an sklearn model saved via ``joblib.save()``
    :return: A model loaded via ``joblib.load()``.
    """
    return joblib.load(path)


def load_dfg(path: Path = ModelPaths.dfg_classifier) -> DFGClassifier:
    """
    Load the :class:`kinactive.model.DFGclassifier`.

    :param path: A path to the saved model. Must contain four directories, for
        `in`, `out`, `other`, and `meta` models.
    :return:
    """
    in_, out, other, meta = (
        load(path / name)
        for name in [
            DumpNames.in_model_dirname + "_classifier",
            DumpNames.out_model_dirname + "_classifier",
            DumpNames.other_model_dirname + "_classifier",
            DumpNames.meta_model_dirname + "_classifier",
        ]
    )
    # meta = load_sklearn(path / (DumpNames.meta_model_dirname + "_classifier"))
    return DFGClassifier(in_, out, other, meta)


def load_kinactive(path: Path = ModelPaths.kinactive_classifier) -> KinactiveClassifier:
    """
    Load the ``KinActive`` model classifying PKs into active/inactive
    conformations.

    :param path: A path to the saved model.
    :return: A loaded model.
    """
    return load(path)


def load(path: Path) -> KinactiveClassifier | KinactiveRegressor | DFGClassifier:
    """
    Automatically determine which model to load based on bath and load this model.

    :param path: A path to the saved model.
    :return: The loaded model.
    """
    if not path.is_dir():
        raise NotADirectoryError(f"{path} must be dir")
    name = path.name.lower()

    if "dfg" in name:
        return load_dfg(path)
    if "classifier" in name:
        cls = KinactiveClassifier
        cls_type = LogisticRegression if "meta" in name else XGBClassifier
    elif "regressor" in name:
        cls = KinactiveRegressor
        cls_type = XGBRegressor
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

    if cls_type is LogisticRegression:
        model = load_sklearn(files[DumpNames.model_filename])
    else:
        model = load_xgb(files[DumpNames.model_filename], cls_type())

    return cls(model, targets, features, params)


if __name__ == "__main__":
    raise RuntimeError
