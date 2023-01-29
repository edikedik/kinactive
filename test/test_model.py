from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from lXtractor.util.io import get_dirs
from sklearn.datasets import make_classification, make_regression
from xgboost import XGBClassifier, XGBRegressor

from kinactive.io import save, load
from kinactive.model import (
    make,
    KinactiveRegressor,
    KinactiveClassifier,
    _generate_stratified_fold_idx,
)


@pytest.fixture
def seq_fs() -> pd.DataFrame:
    return pd.read_csv('data/default_seq_fs.csv')


def load_obj_ids(n: int) -> list[str]:
    df = pd.read_csv('data/default_seq_fs.csv')
    return df['ObjectID'].sample(n).tolist()


def make_data(reg: bool = False, **kwargs) -> pd.DataFrame:
    x, y = make_regression(**kwargs) if reg else make_classification(**kwargs)
    num_y = 1 if len(y.shape) == 1 else y.shape[1]
    df_x = pd.DataFrame(x, columns=[f'X_{i}' for i in range(1, x.shape[1] + 1)])
    df_y = pd.DataFrame(y, columns=[f'Y_{i}' for i in range(1, num_y + 1)])
    df = pd.concat([df_x, df_y], axis=1)
    df['ObjectID'] = load_obj_ids(len(df_x))
    return df


@pytest.mark.parametrize('num_targets', [2, 3])
@pytest.mark.parametrize('is_reg', [False, True])
@pytest.mark.parametrize('starting_params', [{}, {'n_estimators': 20}])
@pytest.mark.parametrize('use_early_stopping', [True, False])
def test_make(num_targets, is_reg, starting_params, use_early_stopping):
    kwargs = dict(n_features=20, n_informative=10)
    if is_reg:
        kwargs['reg'] = True
        kwargs['n_targets'] = num_targets - 1
    else:
        kwargs['n_classes'] = num_targets

    df = make_data(**kwargs)

    target_names = [c for c in df.columns if c.startswith('Y')]
    print(starting_params, use_early_stopping)
    if use_early_stopping:
        starting_params['early_stopping_rounds'] = 3

    model, score, df = make(
        df,
        target_names,
        [],
        starting_params,
        use_early_stopping,
        boruta_kwargs=dict(test_stratify=False, use_test=False),
        classifier=not is_reg,
        n_trials_sel_1=5,
        n_trials_sel_2=5,
    )

    if is_reg:
        assert isinstance(model, KinactiveRegressor)
    else:
        assert isinstance(model, KinactiveClassifier)

    assert score != 0
    assert 0 < len(model.features) < df.shape[1]
    assert len(model.params) > 0
    assert {f'{c}_pred' for c in target_names} == {c for c in df.columns if 'pred' in c}


@pytest.mark.parametrize('is_cls', [True, False])
@pytest.mark.parametrize('params', [{}, {'n_estimators': 10}])
def test_io(is_cls, params, use_early_stopping):
    df = make_data(not is_cls)
    targets = [c for c in df.columns if c.startswith('Y')]
    features = [c for c in df.columns if c.startswith('X')]

    if is_cls:
        model = KinactiveClassifier(
            XGBClassifier(), targets, features, params, use_early_stopping
        )
        suffix = 'classifier'
    else:
        model = KinactiveRegressor(
            XGBRegressor(), targets, features, params, use_early_stopping
        )
        suffix = 'regressor'

    model.train(df)

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        save(model, tmp_dir, 'test')
        dirs = get_dirs(tmp_dir)
        dump_name = f'test_{suffix}'
        assert dump_name in dirs

        model_ = load(tmp_dir / dump_name)
        assert isinstance(model_.model, (XGBRegressor, XGBClassifier))
        assert model.features == model_.features
        assert model.params == model_.params
        assert model.targets == model_.targets


def _get_0_frac(df: pd.DataFrame, col: str = 'Y_1') -> float:
    return (df[col] == 0).sum() / len(df)


@pytest.mark.parametrize('n_classes', [2, 3, 4])
@pytest.mark.parametrize('w0', [0.2, 0.3, 0.4])
@pytest.mark.parametrize('n_folds', [3, 5, 10])
@pytest.mark.parametrize('n_samples', [100, 200, 300])
@pytest.mark.parametrize('n_features', [20])
def test_stratified_fold_index_gen(n_classes, w0, n_folds, n_samples, n_features):
    n_left = n_classes - 1
    w_others = (1 - w0) / n_left
    ws = [w0, *(w_others for _ in range(n_left))]
    assert abs(1 - sum(ws)) < 0.001
    df = make_data(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        weights=ws,
    )
    assert df.shape == (n_samples, n_features + 2)

    actual0_frac = _get_0_frac(df)

    gen_idx = list(
        _generate_stratified_fold_idx(df['ObjectID'].values, df['Y_1'].values, n_folds)
    )
    assert len(gen_idx) == n_folds
    for train_idx, test_idx in gen_idx:
        train_fold, test_fold = df[train_idx], df[test_idx]
        assert len(train_fold) + len(test_fold) == len(df)
        w0_test_frac = _get_0_frac(test_fold)
        w0_train_frac = _get_0_frac(train_fold)
        assert abs(actual0_frac - w0_train_frac) < 0.1
        assert abs(actual0_frac - w0_test_frac) < 0.1
