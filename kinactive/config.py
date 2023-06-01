"""
Configuration dataclasses for the database, matrix and io.
"""
from dataclasses import dataclass
from pathlib import Path

from lXtractor.core.exceptions import MissingData

PK_NAME = "PK"
DFG_MAP = {"in": 0, "out": 1, "other": 2}
DFG_MAP_REV = {v: k for k, v in DFG_MAP.items()}


@dataclass
class DBConfig:
    """
    Database config.

    Default parameters were used to create lXt-PK data collection.
    To reproduce locally, you may change the paths (``*_dir*``) and adjust
    the number of cpus (``*_cpus``).
    """

    #: progress bar output.
    verbose: bool = True

    #: database dump path.
    target_dir: Path = Path("db")
    #: raw PDB structures.
    pdb_dir: Path = Path("pdb") / "structures"
    #: info on PDB structures.
    pdb_dir_info: Path = Path("pdb") / "info"
    #: raw UniProt sequences.
    seq_dir: Path = Path("uniprot") / "fasta"

    #: max trials for fetching an entry from external resources.
    max_fetch_trials: int = 2

    #: #cpus for ``ChainIO`` (10-20 usually works fine).
    io_cpus: int = 1
    #: #cpus for ``ChainInitializer`` (10-20 usually works fine).
    init_cpus: int = 1
    #: #cpus for pairwise sequence alignments. Increase to max number possible.
    init_map_numbering_cpus: int = 1

    #: A path to the PK profile (supplied with the package)
    profile: Path = Path(__file__).parent / "resources" / "Pkinase.hmm"

    #: the domain name to use for extraction.
    pk_map_name: str = PK_NAME
    #: a minimum BitScore to qualify for hit.
    pk_min_score: float = 50
    #: min domain size for canonical sequences.
    pk_min_seq_domain_size: int = 150
    #: min domain size for structure sequences.
    pk_min_str_domain_size: int = 100
    #: min coverage of the hmm nodes.
    pk_min_cov_hmm: float = 0.7
    #: min coverage of the sequence.
    pk_min_cov_seq: float = 0.7
    #: min matching residues' fraction between structure and canonical sequences.
    pk_min_str_seq_match: float = 0.9

    #: minimum sequence size to filter raw sequences from UniProt.
    min_seq_size: int = 150
    #: maximum sequence size to filter raw sequences from UniProt.
    max_seq_size: int = 3000

    #: PDB files format.
    pdb_fmt: str = "cif"
    #: The number of threads to use when fetching data from the PDB.
    pdb_num_fetch_threads: int = 10
    #: The minimum structure size (in residues) to filter raw structures.
    pdb_str_min_size: int = 100

    #: The chunk size to split UniProt ids into when fetching the data from UniProt.
    uniprot_chunk_size: int = 100
    #: The number of threads to use when fetching the data from UniProt.
    uniprot_num_fetch_threads: int = 10

    def __post_init__(self):
        if not self.profile.exists():
            raise MissingData(f"Missing PK profile under {self.profile} path")


@dataclass
class MatrixConfig:
    """
    The superposition-based matrix configuration. This matrix is used to compute
    """

    #: Path to dump the results.
    dir: Path = Path("clustering")

    #: The number of the most covered HMM nodes to use for superposing.
    n_super_pos: int = 30
    #: The PK domain name. Should be the same as used in :class:`kinactive.db.DB`.
    pk_map_name: str = PK_NAME

    #: The number of cpus to use for parallel computation. Adjust carefully.
    n_proc: int | None = None
    #: The chunk size for distributing data between processes.
    chunksize: int = 5000

    #: DFG-Asp/Phe positions.
    df_pos: tuple[int, int] = (141, 142)
    #: Backbone atom names used for superposing.
    bb_atoms: tuple[str, ...] = ("CA",)
    #: DFG-Phe atom names used for RMSD computation.
    phe_atoms: tuple[str, ...] = ("CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    #: DFG-Asp atom names used for RMSD computation.
    asp_atoms: tuple[str, ...] = ("CA", "CB", "CG", "OD1", "OD2")


@dataclass
class _DumpNames:
    cls_keyword: str = "classifier"
    reg_keyword: str = "regressor"

    model_filename: str = "model.bin"
    features_filename: str = "features.txt"
    targets_filename: str = "targets.txt"
    params_filename: str = "params.json"

    in_model_dirname: str = "in"
    out_model_dirname: str = "out"
    other_model_dirname: str = "other"
    meta_model_dirname: str = "meta"

    positions_ca: str = "positions_CA.txt"
    distances: str = "distances.csv"

    summary_parent_seq = "initial_seq_summary.csv"
    summary_parent_str = "initial_str_summary.csv"
    summary_child_seq = "domain_seq_summary.csv"
    summary_child_str = "domain_str_summary.csv"

    canonical_seq_vs = "defaults_can_seq_vs.csv"
    structure_seq_vs = "defaults_str_seq_vs.csv"
    ligand_vs = "default_lig_vs.csv"
    structure_vs = "default_str_vs.csv"

    @property
    def summary_file_names(self) -> tuple[str, str, str, str]:
        return (
            self.summary_parent_seq,
            self.summary_parent_str,
            self.summary_child_seq,
            self.summary_child_str,
        )


@dataclass
class _ModelPaths:
    base: Path = Path(__file__).parent / "resources" / "models"
    kinactive_classifier: Path = base / "kinactive_classifier"
    dfg_classifier: Path = base / "DFG_classifier"


@dataclass
class _ColNames:
    dfg: str = "DFG"
    dfg_manual: str = "DFG_manual"
    dfg_pred = "DFG_pred"
    dfg_cls: str = "DFG_cls"
    dfg_cls_pred: str = "DFG_cls_pred"
    is_dfg_in: str = "is_DFG_in"
    is_dfg_out: str = "is_DFG_out"
    is_dfg_other: str = "is_DFG_other"
    dfg_in_proba: str = "in_proba"
    dfg_out_proba: str = "out_proba"
    dfg_other_proba: str = "other_proba"
    dfg_in_meta_prob: str = "in_meta_proba"
    dfg_out_meta_prob: str = "out_meta_prob"
    dfg_other_meta_prob: str = "other_meta_prob"

    rmsd_ca: str = "RMSD_CA"
    rmsd_df: str = "RMSD_DF"
    id_fix: str = "ID_fix"
    id_mob: str = "ID_mob"

    @property
    def is_dfg_cols(self) -> tuple[str, str, str]:
        return self.is_dfg_in, self.is_dfg_out, self.is_dfg_other

    @property
    def dfg_proba_cols(self) -> tuple[str, str, str]:
        return self.dfg_in_proba, self.dfg_out_proba, self.dfg_other_proba

    @property
    def dfg_meta_proba_cols(self) -> tuple[str, str, str]:
        return self.dfg_in_meta_prob, self.dfg_out_meta_prob, self.dfg_other_meta_prob

    @property
    def dfg_cols(self) -> list[str]:
        return [
            self.dfg,
            self.dfg_pred,
            self.dfg_cls,
            self.dfg_cls_pred,
            *self.is_dfg_cols,
            *self.dfg_proba_cols,
            *self.dfg_meta_proba_cols,
        ]


DumpNames = _DumpNames()
ColNames = _ColNames()
ModelPaths = _ModelPaths()
DefaultMatrixConfig = MatrixConfig()

if __name__ == "__main__":
    raise RuntimeError
