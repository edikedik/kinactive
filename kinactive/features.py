"""
Variables' definitions and calculation.
"""
import logging
import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path

import pandas as pd
from lXtractor.core.chain import ChainSequence, ChainStructure, Chain, ChainList
from lXtractor.util import get_files
from lXtractor.variables import (
    GenericCalculator,
    Manager,
    SeqEl,
    PFP,
    Dist,
    PseudoDihedral,
    Phi,
    Psi,
    Chi1,
    SASA,
    LigandDist,
    LigandNames,
    LigandContactsCount,
    AggDist,
)
from lXtractor.variables.base import StructureVariable, SequenceVariable
from more_itertools import windowed

from kinactive.config import PK_NAME, DumpNames

CT: t.TypeAlias = ChainSequence | ChainStructure
VT: t.TypeAlias = SequenceVariable | StructureVariable

LOGGER = logging.getLogger(__name__)
Results = t.NamedTuple(
    "Results",
    [
        ("seq_vs", pd.DataFrame),
        ("str_seq_vs", pd.DataFrame),
        ("lig_vs", pd.DataFrame),
        ("str_vs", pd.DataFrame),
    ],
)
PocketPos = (
    7,
    8,
    9,
    10,
    14,
    15,
    28,
    30,
    48,
    61,
    77,
    78,
    79,
    80,
    81,
    83,
    84,
    87,
    123,
    125,
    127,
    128,
    130,
    140,
    141,
    142,
)


def _make_pdist(ps1, ps2, a1="CB", a2="CB", min_sep=1):
    if ps2 is None:
        for i, j in combinations(ps1, 2):
            if abs(i - j) >= min_sep:
                yield Dist(i, j, a1, a2)
    else:
        for i in ps1:
            for j in ps2:
                if abs(i - j) >= min_sep:
                    yield Dist(i, j, a1, a2)


def _make_pdist_agg(ps1, ps2, key, min_sep=1):
    if ps2 is None:
        for i, j in combinations(ps1, 2):
            if abs(i - j) >= min_sep:
                yield AggDist(i, j, key)
    else:
        for i in ps1:
            for j in ps2:
                if abs(i - j) >= min_sep:
                    yield AggDist(i, j, key)


def _make_pfp(ps, n_comp: int = 3):
    for p in ps:
        for i in range(1, n_comp + 1):
            yield PFP(p, i)


@dataclass
class DefaultFeatures:
    """
    A default feature set based on the PF00069 PK profile positions.
    """

    #: PK HMM profile positions
    profile_pos: tuple[int, ...] = tuple(range(1, 265))
    #: xDFGx profile positions (DFG motif plus the two residues around).
    xdfgx: tuple[int, ...] = tuple(range(140, 144))
    #: HRD motif's profile positions.
    hrd: tuple[int, int, int] = (121, 122, 123)
    #: Activation loop profile positions.
    al: tuple[int, ...] = tuple(range(135, 150))
    #: B3 sheet profile positions.
    b3_sheet: tuple[int, ...] = tuple(range(24, 31))
    #: aC helix profile positions.
    ac_helix: tuple[int, ...] = tuple(range(37, 57))
    #: Pocket residues profile positions.
    pocket: tuple[int, ...] = PocketPos

    def make_str_vs(self) -> tuple[StructureVariable, ...]:
        """
        Make a list of structural variables including::

            #. SASA for each position.
            #. Pseudo dihedral angles for each consecutive quadruplet.
            #. Phi angles for each position except the very first one.
            #. Psi angles for each position except the very last one.
            #. Chi1 angles for each position.
            #. Pairwise CB-CB distances between the pocket residues.
            #. Distances from the pocket residues CB to the DFG-Asp CG atom.
            #. Distances from the pocket residues CB to the DFG-Phe CZ atom.
            #. A distance between the DFG-Asp CG and the DFG-Phe CZ
            #. A distance between the B3-Lys NZ and aC-Glu CD

        :return: A default set of structural variables.
        """
        profile_pos = self.profile_pos
        return tuple(
            chain(
                (SASA(x) for x in profile_pos),
                (PseudoDihedral(*x) for x in windowed(profile_pos, 4)),
                (Phi(x) for x in profile_pos[1:]),
                (Psi(x) for x in profile_pos[:-1]),
                (Chi1(x) for x in profile_pos),
                _make_pdist(self.pocket, [141], a2="CG"),
                _make_pdist(self.pocket, [142], a2="CZ"),
                _make_pdist(self.pocket, None),
                [Dist(141, 142, "CG", "CZ"), Dist(30, 48, "NZ", "CD")],
            )
        )

    def make_lig_vs(self) -> tuple[StructureVariable, ...]:
        """
        Make a default list of ligand variables including::

            #. A count of ligand contacts per position.
            #. A minimum position-wise distance to the closest ligand.
            #. The closest contacting ligand's name per position.

        :return: A default set of ligand variables.
        """
        pos = self.profile_pos
        return tuple(
            chain(
                (LigandContactsCount(p) for p in pos),
                (LigandDist(p) for p in pos),
                (LigandNames(p) for p in pos),
            )
        )

    def make_seq_vs(self) -> tuple[SequenceVariable, ...]:
        """
        Make a default list of sequence variables including::

            #. Sequence elements at positions 30, 48 and 140-144
            #. ProtFP variables with three components for each profile position.

        :return: A default set of sequence variables.
        """
        return (
            SeqEl(30),
            SeqEl(48),
            *(SeqEl(x) for x in self.xdfgx),
            *_make_pfp(self.profile_pos),
        )

    def calculate_seq_vs(
        self,
        chains: abc.Sequence[ChainSequence],
        map_name: str = PK_NAME,
        num_proc: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate default sequence variables.

        :param chains: A sequence of chain sequences.
        :param map_name: A reference name.
        :param num_proc: The number of CPUs to use.
        :param verbose: Display progress bar.
        :return: A table with calculated variables.
        """
        return calculate(
            chains, self.make_seq_vs(), num_proc, verbose, map_name=map_name
        )

    def calculate_str_vs(
        self,
        chains: abc.Sequence[ChainStructure],
        map_name: str = PK_NAME,
        num_proc: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate default structure variables.

        :param chains: A sequence of chain structures.
        :param map_name: A reference name.
        :param num_proc: The number of CPUs to use.
        :param verbose: Display progress bar.
        :return: A table with calculated variables.
        """
        return calculate(
            chains, self.make_str_vs(), num_proc, verbose, map_name=map_name
        )

    def calculate_lig_vs(
        self,
        chains: abc.Sequence[ChainStructure],
        map_name: str = PK_NAME,
        num_proc: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate default ligand variables.

        :param chains: A sequence of chain structures.
        :param map_name: A reference name.
        :param num_proc: The number of CPUs to use.
        :param verbose: Display progress bar.
        :return: A table with calculated variables.
        """
        return calculate(
            chains, self.make_lig_vs(), num_proc, verbose, map_name=map_name
        )

    def calculate_all_vs(
        self,
        chains: abc.Sequence[Chain],
        map_name: str = PK_NAME,
        num_proc: int | None = None,
        verbose: bool = True,
        base: Path | None = None,
        overwrite: bool = False,
    ) -> Results:
        """
        Calculate default variables. These include four sets::

            #. A default set of sequence variables for canonical sequences.
            #. A default set of sequence variables for structure sequences.
            #. A default set of structure variables.
            #. A default set of ligand variables.

        :param chains: A sequence of chains.
        :param map_name: A reference name.
        :param num_proc: The number of CPUs to use.
        :param verbose: Display progress bar.
        :param base: Base path to save the results to. If not provided, the
            results are returned but not saved.
        :param overwrite: Overwrite existing files. If False, will skip the
            calculation of existing variables.
        :return: A named tuple with calculated variables' tables.
        """
        if not isinstance(chains, ChainList):
            chains = ChainList(chains)
        kw = {"map_name": map_name, "verbose": verbose}

        staged = [
            (
                self.calculate_seq_vs,
                chains.sequences,
                "canonical seqs",
                DumpNames.canonical_seq_vs,
            ),
            (
                self.calculate_seq_vs,
                chains.structure_sequences,
                "structure seqs",
                DumpNames.structure_seq_vs,
            ),
            (
                self.calculate_lig_vs,
                chains.structures,
                "ligand variables",
                DumpNames.ligand_vs,
            ),
            (
                self.calculate_str_vs,
                chains.structures,
                "structure variables",
                DumpNames.structure_vs,
            ),
        ]
        dfs = []
        for meth, objs, desc, name in staged:
            if "seqs" in desc:
                desc = f"Calculating sequence variables on {desc}"
            else:
                desc = f"Calculating {desc}"
            # Use multiprocessing only for structure variables
            if "structure variables" in desc:
                kw["num_proc"] = num_proc
            # Check if already exists
            df = None
            if base is not None and base.exists():
                files = get_files(base)
                if name in files and not overwrite:
                    LOGGER.info(f"Reading already calculated {name}")
                    df = pd.read_csv(files[name])
            # Calculate if wasn't loaded
            if df is None:
                LOGGER.info(desc)
                df = meth(objs, **kw)
                LOGGER.info(f"Resulting shape: {df.shape}")
            dfs.append(df)
            # Save if base path was provided
            if base is not None:
                base.mkdir(exist_ok=True)
                df.to_csv(base / name, index=False)
                LOGGER.info(f"Saved {name} to {base}")

        LOGGER.info("Finished calculations")

        return Results(*dfs)


def calculate(
    chains: abc.Sequence[CT],
    vs: abc.Sequence[VT],
    num_proc: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate variables and aggregate the results.

    :param chains: A sequence of `Chain*`-type objects.
    :param vs: A sequence of variables to calculate.
    :param num_proc: A number of processors to use for the calculation.
    :param verbose: Display progress bar.
    :param kwargs: Passed to the :meth:`Manager.calculate`
        (see ``lXtractor`` docs).
    :return:
    """
    manager = Manager(verbose=verbose)
    calculator = GenericCalculator(num_proc=num_proc)
    results = manager.calculate(chains, vs, calculator, **kwargs)
    df = manager.aggregate_from_it(results, replace_errors=True)
    # assert isinstance(df, pd.DataFrame), 'Failed to convert results into a table'
    return df


if __name__ == "__main__":
    raise RuntimeError
