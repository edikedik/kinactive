from collections import abc
from dataclasses import dataclass
from pathlib import Path

from lXtractor.core.base import SOLVENTS
from lXtractor.core.exceptions import MissingData

PK_NAME = 'PK'


@dataclass
class DBConfig:
    verbose: bool

    target_dir: Path
    pdb_dir: Path
    seq_dir: Path
    pdb_dir_info: Path | None

    max_fetch_trials: int = 2

    init_cpus: int | None = None
    io_cpus: int | None = None
    init_map_numbering_cpus: int | None = None

    profile: Path = Path(__file__).parent / 'resources' / 'Pkinase.hmm'

    pk_map_name: str = PK_NAME
    pk_min_score: float = 30
    pk_min_seq_domain_size: int = 150
    pk_min_str_domain_size: int = 100
    pk_min_cov_hmm: float = 0.7
    pk_min_cov_seq: float = 0.7
    pk_min_str_seq_match: float = 0.7

    min_seq_size: int = 150
    max_seq_size: int = 3000

    pdb_fmt: str = 'cif'
    pdb_num_fetch_threads: int = 10
    pdb_str_min_size: int = 100
    pdb_solvents: abc.Sequence[str] = SOLVENTS

    uniprot_chunk_size: int = 100
    uniprot_num_fetch_threads: int = 10

    def __post_init__(self):
        if not self.profile.exists():
            raise MissingData(f'Missing PK profile under {self.profile} path')


@dataclass
class _DumpNames:
    cls_keyword: str = 'classifier'
    reg_keyword: str = 'regressor'

    model_filename: str = 'model.bin'
    features_filename: str = 'features.txt'
    targets_filename: str = 'targets.txt'
    params_filename: str = 'params.json'

    in_model_dirname: str = 'in'
    out_model_dirname: str = 'out'
    inter_model_dirname: str = 'inter'
    meta_model_dirname: str = 'meta'


@dataclass
class _ModelPaths:
    kinactive_classifier: Path = (
        Path(__file__).parent / 'resources' / 'models' / 'kinactive_classifier'
    )


if __name__ == '__main__':
    raise RuntimeError
