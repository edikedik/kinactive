import sys
import typing as t
from collections import abc
from itertools import chain
from pathlib import Path
from shutil import unpack_archive

import click
import lXtractor.chain as lxc
import pandas as pd
from lXtractor.util import fetch_to_file
from loguru import logger as LOGGER
from more_itertools import ilen
from toolz import curry

from kinactive.config import ColNames, load_data_links
from kinactive.db import DB
from kinactive.features import (
    load_seq_model_features,
    calculate,
    load_str_model_features,
)
from kinactive.io import load_seq_models, load_str_models
from kinactive.parsers import SequenceParser, StructureParser

_DefaultFlagKwsTrue = dict(is_flag=True, default=True, show_default=True)
_DefaultFlagKwsFalse = dict(is_flag=True, default=False, show_default=True)
_StructureFormats = ("pdb", "cif", "mmtf", "pdb.gz", "cif.gz", "mmtf.gz")

_CS = t.TypeVar("_CS", lxc.ChainSequence, lxc.ChainStructure)

LOGGER.remove()


@click.group(
    "kinactive",
    context_settings=dict(
        help_option_names=["-h", "--help"], ignore_unknown_options=True
    ),
    no_args_is_help=True,
    invoke_without_command=True,
)
def kinactive():
    """
    KinActive: a CLI tool for protein kinase structures and sequences.

    Author   : Ivan Reveguk.

    Email    : ivan.reveguk@gmail.com
    """
    pass


@kinactive.command("fetch", no_args_is_help=True)
@click.option(
    "--db",
    **_DefaultFlagKwsFalse,
    help="lXtractor data collection of PK sequences and structures.",
)
@click.option(
    "--pdb_struc_pred",
    **_DefaultFlagKwsFalse,
    help="DFG and active/inactive predictions for all structural domains in the lXt-PK "
    "data collection produced by DFGclassifier and KinActive models.",
)
@click.option(
    "--pdb_struc_features",
    **_DefaultFlagKwsFalse,
    help="Default feature set for all structural domains in the lXt-PK data "
    "collection.",
)
@click.option(
    "--pdb_lig_features",
    **_DefaultFlagKwsFalse,
    help="Default feature set of ligand variables for all structural domain in the "
    "lXt-PK data collection.",
)
@click.option(
    "--pdb_seq_features",
    **_DefaultFlagKwsFalse,
    help="Default feature set for all domain structure sequences in the lXt-PK "
    "data collection.",
)
@click.option(
    "--pdb_can_seq_features",
    **_DefaultFlagKwsFalse,
    help="Default feature set for domains of all canonical UniProt sequences "
    "encompassed by the lXt-PK data collection.",
)
@click.option(
    "--sp_seq_pred",
    **_DefaultFlagKwsFalse,
    help="Predictions of DFG-in/DFG-out conformational propensities for all PK domains "
    "found in SwissProt.",
)
@click.option(
    "--sp_model_seq_features",
    **_DefaultFlagKwsFalse,
    help="A set of sequence variables necessary to run all sequence-based models for "
    "all PK domains in SwissProt.",
)
@click.option(
    "-a", "--all_data", **_DefaultFlagKwsFalse, help="Download all available data."
)
@click.option(
    "-u",
    "--unpack",
    **_DefaultFlagKwsFalse,
    help="Unpack each downloaded tar.gz archive.",
)
@click.option(
    "-r",
    "--rm_unpacked",
    **_DefaultFlagKwsFalse,
    help="Remove fetched archive after unpacking.",
)
@click.option("-l", "--links", **_DefaultFlagKwsFalse, help="Display data links.")
@click.option(
    "-o",
    "--out_dir",
    help="A path to a directory where to save files. If not provided, the current "
    "working directory will be used.",
)
@click.option(
    "-v",
    "--verbose",
    **_DefaultFlagKwsFalse,
    help="Output basic logging information to stdout.",
)
def fetch(
    db,
    pdb_struc_pred,
    pdb_struc_features,
    pdb_lig_features,
    pdb_seq_features,
    pdb_can_seq_features,
    sp_seq_pred,
    sp_model_seq_features,
    all_data,
    unpack,
    rm_unpacked,
    links,
    out_dir,
    verbose,
):
    """
    A simple utility to fetch related data.

    All options except `out_dir` are flags.

    Usage example: `kinactive fetch --pdb_struc_pred --sp_seq_pred -vru` will
    fetch ML model predictions for SwissProt sequences and PDB structures,
    unpack fetched archives, remove them, and output basic logging info.

    To fetch the data manually, navigate to
    https://zenodo.org/doi/10.5281/zenodo.10256947
    """
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        LOGGER.add(sys.stdout)

    data_links = load_data_links()

    if links:
        for k, v in data_links.items():
            if k == "version":
                print(f"Version: {v}")
                print('Option\tLink')
            else:
                print(f"{k}\t{v}")

    targets = [
        ("db", db, "lXt-PK database"),
        (
            "pdb_struc_pred",
            pdb_struc_pred,
            "structure-based predictions for PDB models",
        ),
        ("pdb_struc_features", pdb_struc_features, "PDB domains' structural features"),
        ("pdb_lig_features", pdb_lig_features, "PDB domains' ligand-related features"),
        ("pdb_seq_features", pdb_seq_features, "PDB domains' sequence features"),
        (
            "pdb_can_seq_features",
            pdb_can_seq_features,
            "PDB domains' features for related canonical seqs",
        ),
        ("sp_seq_pred", sp_seq_pred, "predictions for SwissProt sequences"),
        (
            "sp_model_seq_features",
            sp_model_seq_features,
            "features for SwissProt sequences",
        ),
    ]
    for t_name, t_arg, t_desc in targets:
        if t_arg or all_data:
            LOGGER.info(f"Fetching {t_desc}.")
            t_path = fetch_to_file(data_links[t_name], root_dir=out_dir)
            if unpack:
                LOGGER.info(f"Unpacking {t_path}")
                unpack_archive(t_path, out_dir, "gztar")
                if rm_unpacked:
                    t_path.unlink()
            # LOGGER.info(f"Finished fetching {t_desc}")

    LOGGER.info("Done fetching data")


@kinactive.command("db", no_args_is_help=True)
@click.option("-c", "--config", type=click.Path(dir_okay=False, file_okay=True))
@click.option("-o", "--overwrite", **_DefaultFlagKwsFalse)
def db(config, overwrite):
    """
    Construct or update a new database.

    (!) TBD: The implementation is subject to lXtractor update.
    """
    pass


@kinactive.command("predict", no_args_is_help=True)
@click.argument("inputs", nargs=-1)
@click.option(
    "-t",
    "--inp_type",
    type=click.types.Choice(["s", "S", "a"]),
    required=True,
    help="Input(s) type. "
    "s -> Sequence input. If a file is provided, only the first sequence "
    "is considered. "
    "S -> Structure input. See `pdb_fmt` option for supported formats. "
    "a -> AlphaFold input. Should only be used if inputs are identifiers "
    "in the AlphaFold2 database (such as UniProt accessions).",
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True, file_okay=False),
    help="A path to an output directory. If provided, the sequence or structure "
    "data collection, calculated variables and predictions will be saved to this dir.",
)
@click.option(
    "-d",
    "--domains",
    is_flag=True,
    help="A flag signifying that the inputs are paths to already extracted chain "
    "sequence or chain structure domains. Thus, they already exist and won't be saved "
    "separately if the `out_dir` is provided.",
)
@click.option(
    "-V",
    "--variables",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="A path to already calculated variables -- an existing CSV file that can be "
    "loaded into a pandas dataframe and contains variables required for prediction. "
    "If provided, the inputs will be ignored.",
)
@click.option(
    "--pdb_fmt",
    type=click.types.Choice(_StructureFormats),
    default="mmtf.gz",
    show_default=True,
    help="PDB file format to fetch.",
)
@click.option(
    "--af2_fmt",
    type=click.types.Choice(_StructureFormats[:2]),
    default="cif",
    show_default=True,
    help="A structure file format to fetch from the AlphaFold DB.",
)
@click.option(
    "--str_out_fmt",
    type=click.types.Choice(_StructureFormats),
    default="mmtf.gz",
    show_default=True,
    help="A structure file format to save parsed structures in.",
)
@click.option(
    "-A",
    "--altloc",
    is_flag=True,
    default=False,
    show_default=True,
    help="Split alternative locations into separate entities if they are available. "
    "The atoms without specified altloc will be distributed (copied) into "
    "alternative structures.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Display logging information and progress bars.",
)
@click.option(
    "-n",
    "--num_proc",
    type=int,
    default=1,
    show_default=True,
    help="The number of CPUs to use. Increasing makes sense when processing large "
    "number of structures. Not beneficial for sequences.",
)
@click.option(
    "-l",
    "--log_file",
    is_flag=True,
    help="Save logging information into file. If `out_dir` is provided, will create "
    "a `logs` subdir to save files to.",
)
def predict(
    inputs,
    inp_type,
    out_dir,
    domains,
    variables,
    pdb_fmt,
    af2_fmt,
    str_out_fmt,
    altloc,
    verbose,
    num_proc,
    log_file,
):
    """
    Predict labels for a small number of sequences or structures.
    For large-scale applications, one should first compile an `lXtractor`
    data collection (see `db` command).

    Inputs are provided as arguments and can be of two kinds:

    1. A path to an existing sequence or structure.

    2. An external resource identifier (PDB, UniProt, or AlphaFold).
        In that case, the data will be fetched from that resource.

    The input type (`-t` or `--inp_type`) must be specified (see the docs below).

    """
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        LOGGER.add(sys.stdout)
        if log_file:
            if out_dir:
                log_dir = out_dir / "logs"
                log_dir.mkdir(exist_ok=True, parents=True)
                LOGGER.add(log_dir / "file_{time}.log")
            else:
                LOGGER.add("file_{time}.log")

    if variables is not None:
        vs = pd.read_csv(variables)
        LOGGER.info(
            f"Loaded {vs.shape[1] - 1} existing variables for {len(vs)} objects."
        )
        dfp = predict_on_seqs(vs) if inp_type == "s" else predict_on_strs(vs)

        if out_dir:
            dfp.to_csv(out_dir / "predictions.csv", index=False)
            LOGGER.info(f"Saved predictions to {out_dir}")

        _print_df(dfp)
        return

    LOGGER.info(f"Preparing {len(inputs)} inputs.")
    chains = lxc.ChainList(prepare_inputs(inputs, inp_type, pdb_fmt, af2_fmt, altloc))
    LOGGER.info(f"Initialized {len(chains)} chains.")

    if domains:
        extracted = True
        domains = chains
    else:
        extracted = False
        _db = DB()
        chains = _db.discover_domains(chains)
        domains = chains.collapse_children()

    LOGGER.info(
        f"Discovered {len(domains)} domains; "
        f"{len(chains)} domain-containing chains remained."
    )
    if len(domains) == 0:
        LOGGER.warning("No domains discovered; terminating...")
        return

    if inp_type in ["S", "a"]:
        vs = calculate(
            domains, load_str_model_features(), num_proc, verbose, map_name="PK"
        )
        LOGGER.info(
            f"Calculated {vs.shape[1] - 1} structure features for {len(vs)} domains"
        )
        dfp = predict_on_strs(vs)
    else:
        vs = calculate(
            domains, load_seq_model_features(), num_proc, verbose, map_name="PK"
        )
        dfp = predict_on_seqs(vs)

    if out_dir:
        if extracted:
            LOGGER.info("The domains were already extracted; chains won't be saved.")
        else:
            chains_dir = out_dir / "chains"
            chains_dir.mkdir(exist_ok=True, parents=True)
            kw = dict(write_children=True)
            if inp_type in ["S", "a"]:
                kw["fmt"] = str_out_fmt
            io = lxc.ChainIO(verbose=verbose, num_proc=num_proc)
            num_saved = ilen(io.write(chains, out_dir / "chains", **kw))
            LOGGER.info(f"Wrote {num_saved} chains to {chains_dir}.")

        vs.to_csv(out_dir / "variables.csv", index=False)
        LOGGER.info(f"Saved calculated variables to {out_dir}.")

        dfp.to_csv(out_dir / "predictions.csv", index=False)
        LOGGER.info(f"Saved predictions to {out_dir}.")

    _print_df(dfp)


def _print_df(df: pd.DataFrame) -> None:
    stream = click.get_text_stream("stdout")
    df.to_string(stream, index=False)
    stream.write("\n")


def prepare_inputs(
    inputs: abc.Sequence[str], inp_type: str, pdb_fmt: str, af2_fmt: str, altloc: bool
) -> abc.Iterator[_CS]:
    if inp_type == "s":
        parser = SequenceParser()
        objs = map(parser, inputs)
    elif inp_type in ["S", "a"]:
        parser = StructureParser(pdb_fmt=pdb_fmt, af2_fmt=af2_fmt, split_altloc=altloc)
        alphafold = inp_type == "a"
        objs = chain.from_iterable(map(curry(parser)(alphafold=alphafold), inputs))
    else:
        raise RuntimeError("...")
    yield from objs


def predict_on_seqs(vs: pd.DataFrame) -> pd.DataFrame:
    vs = vs.copy()
    models = load_seq_models()

    tk_prob = models["TkST"].predict_proba(vs)[:, 1]
    is_tk = tk_prob > 0.5
    is_stk = ~is_tk
    vs["TKprob"] = tk_prob
    vs["PredictedGroup"] = pd.Series(is_tk).map({False: "STK", True: "TK"})

    if is_tk.any():
        vs.loc[is_tk, [ColNames.ahao_prob_col]] = models["AHAO_Tk"].predict_proba(
            vs.loc[is_tk]
        )[:, 1]
        vs.loc[is_tk, ColNames.aaio_prob_col] = models["AAIO_Tk"].predict_proba(
            vs.loc[is_tk]
        )[:, 1]
    if is_stk.any():
        vs.loc[is_stk, ColNames.ahao_prob_col] = models["AHAO_STk"].predict_proba(
            vs.loc[is_stk]
        )[:, 1]
        vs.loc[is_stk, ColNames.aaio_prob_col] = models["AAIO_STk"].predict_proba(
            vs.loc[is_stk]
        )[:, 1]

    vs[ColNames.ahao_col] = vs[ColNames.ahao_prob_col].round().astype(bool)
    vs[ColNames.aaio_col] = vs[ColNames.aaio_prob_col].round().astype(bool)

    return vs[[c for c in vs.columns if "(" not in c]]


def predict_on_strs(vs: pd.DataFrame) -> pd.DataFrame:
    vs = vs.copy()

    models = load_str_models()
    vs["Active"] = models["kinactive"].predict(vs).astype(bool)
    vs = models["DFG"].predict_full(vs)

    return vs[[c for c in vs.columns if "(" not in c]]


if __name__ == "__main__":
    kinactive()
