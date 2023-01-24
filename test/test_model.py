from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from lXtractor.util.io import get_dirs
from sklearn.datasets import make_classification, make_regression
from xgboost import XGBClassifier, XGBRegressor

from kinactive.model import make, KinactiveRegressor, KinactiveClassifier, save, load


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
def test_make(num_targets, is_reg, starting_params):
    kwargs = dict(n_features=20, n_informative=10)
    if is_reg:
        kwargs['reg'] = True
        kwargs['n_targets'] = num_targets - 1
    else:
        kwargs['n_classes'] = num_targets

    df = make_data(**kwargs)

    target_names = [c for c in df.columns if c.startswith('Y')]

    model, score = make(
        df,
        target_names,
        starting_params,
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


@pytest.mark.parametrize('is_cls', [True, False])
@pytest.mark.parametrize('params', [{}, {'n_estimators': 5}])
def test_io(is_cls, params):
    df = make_data(not is_cls)
    targets = [c for c in df.columns if c.startswith('Y')]
    features = [c for c in df.columns if c.startswith('X')]

    if is_cls:
        model = KinactiveClassifier(XGBClassifier(), targets, features, params)
        suffix = 'classifier'
    else:
        model = KinactiveRegressor(XGBRegressor(), targets, features, params)
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
