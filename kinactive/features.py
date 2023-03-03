import logging
import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import chain

import pandas as pd
from lXtractor.core.chain import ChainSequence, ChainStructure, Chain, ChainList
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
)
from lXtractor.variables.base import StructureVariable, SequenceVariable
from more_itertools import windowed

from kinactive.config import PK_NAME

# CT = t.TypeVar('CT', ChainSequence, ChainStructure)
CT: t.TypeAlias = ChainSequence | ChainStructure
VT: t.TypeAlias = SequenceVariable | StructureVariable

LOGGER = logging.getLogger(__name__)
Results = t.NamedTuple(
    "Results",
    [
        ("seq_vs", pd.DataFrame),
        ("str_seq_vs", pd.DataFrame),
        ("str_vs", pd.DataFrame),
        ("lig_vs", pd.DataFrame),
    ],
)


def _make_pdist(ps1, ps2, a1="CB", a2="CB"):
    for i in ps1:
        for j in ps2:
            yield Dist(i, j, a1, a2)


def _make_pfp(ps, n_comp: int = 3):
    for p in ps:
        for i in range(1, n_comp + 1):
            yield PFP(p, i)


@dataclass
class DefaultFeatures:
    """
    Default feature set is based on PF00069 Pfam profile positions.
    It includes:

        #. SASA for each position.
        #. Pseudo dihedral angles for each consecutive quadruplet.
        #. Phi angles for each position except the very first one.
        #. Psi angles for each position except the very last one.
        #. Chi1 angles for each position.
        #. Pairwise CB-CB distances between:

            #. $\beta_3$-sheet and $\alpha C$-helix.
            #. $\beta_3$-sheet and XDFGX region.
            #. $\alpha C$-helix and XDFGX region.

        #. $DFG_{Phe}-C\zeta$ -- $\alphaC_{Glu+4}-C\alpha$ distance.
        #. $DFG_{Phe}-C\zeta$ -- $\beta_3_{Lys}-C\alpha$ distance.
    """

    xdfgx: tuple[int, ...] = tuple(range(140, 145))
    b3_sheet: tuple[int, ...] = tuple(range(24, 31))
    ac_helix: tuple[int, ...] = tuple(range(37, 57))
    profile_pos: tuple[int, ...] = tuple(range(1, 265))

    def make_str_vs(self) -> tuple[StructureVariable, ...]:
        profile_pos = self.profile_pos
        return tuple(
            chain(
                (SASA(x) for x in profile_pos),
                (PseudoDihedral(*x) for x in windowed(profile_pos, 4)),
                (Phi(x) for x in profile_pos[1:]),
                (Psi(x) for x in profile_pos[:-1]),
                (Chi1(x) for x in profile_pos),
                chain(
                    _make_pdist(self.b3_sheet, self.ac_helix),
                    _make_pdist(self.b3_sheet, self.xdfgx),
                    _make_pdist(self.ac_helix, self.xdfgx),
                    [
                        Dist(142, 52, "CZ", "CA"),
                        Dist(142, 30, "CZ", "CA"),
                    ],
                ),
            )
        )

    def make_lig_vs(self) -> tuple[StructureVariable, ...]:
        pos = self.profile_pos
        return tuple(
            chain(
                (LigandContactsCount(p) for p in pos),
                (LigandDist(p) for p in pos),
                (LigandNames(p) for p in pos),
            )
        )

    def make_seq_vs(self) -> tuple[SequenceVariable, ...]:
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
        return calculate(
            chains, self.make_lig_vs(), num_proc, verbose, map_name=map_name
        )

    def calculate_all_vs(
        self,
        chains: abc.Sequence[Chain],
        map_name: str = PK_NAME,
        num_proc: int | None = None,
        verbose: bool = True,
    ) -> Results:
        if not isinstance(chains, ChainList):
            chains = ChainList(chains)
        kw = {"map_name": map_name, "num_proc": num_proc, "verbose": verbose}

        LOGGER.info("Calculating sequence variables on canonical seqs")
        df_can_seq = self.calculate_seq_vs(chains.sequences, **kw)
        LOGGER.info(f"Canonical sequences features shape: {df_can_seq.shape}")

        LOGGER.info("Calculating sequence variables on structure seqs")
        df_str_seq = self.calculate_seq_vs(chains.structure_sequences, **kw)
        LOGGER.info(f"Structure sequences features shape: {df_str_seq.shape}")

        LOGGER.info("Calculating structure variables")
        df_str = self.calculate_str_vs(chains.structures, **kw)
        LOGGER.info(f"Structure features shape: {df_str.shape}")

        LOGGER.info("Calculating ligand variables")
        df_lig = self.calculate_lig_vs(chains.structures, **kw)
        LOGGER.info(f"Ligand features shape: {df_lig.shape}")

        LOGGER.info("Finished calculations")

        return Results(df_can_seq, df_str_seq, df_str, df_lig)


def calculate(
    chains: abc.Sequence[CT],
    vs: abc.Sequence[VT],
    num_proc: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    manager = Manager(verbose=verbose)
    calculator = GenericCalculator(num_proc=num_proc)
    results = manager.calculate(chains, vs, calculator, **kwargs)
    df = manager.aggregate_from_it(results, replace_errors=True)
    # assert isinstance(df, pd.DataFrame), 'Failed to convert results into a table'
    return df


if __name__ == "__main__":
    raise RuntimeError
