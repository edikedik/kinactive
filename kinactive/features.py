import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import chain

import pandas as pd
from lXtractor.core.chain import ChainSequence, ChainStructure
from lXtractor.variables.base import StructureVariable, SequenceVariable
from lXtractor.variables.calculator import GenericCalculator
from lXtractor.variables.manager import Manager
from lXtractor.variables.sequential import SeqEl, PFP
from lXtractor.variables.structural import Dist, PseudoDihedral, Phi, Psi, Chi1, SASA
from more_itertools import windowed

# CT = t.TypeVar('CT', ChainSequence, ChainStructure)
CT: t.TypeAlias = ChainSequence | ChainStructure
VT: t.TypeAlias = SequenceVariable | StructureVariable


def _make_pdist(ps1, ps2, a1='CB', a2='CB'):
    for i in ps1:
        for j in ps2:
            yield Dist(i, j, a1, a2)


def _make_pfp(ps, n_comp: int = 3):
    for p in ps:
        for i in range(1, n_comp + 1):
            yield PFP(p, i)


@dataclass
class DefaultFeatures:
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
                        Dist(142, 52, 'CZ', 'CA'),
                        Dist(142, 30, 'CZ', 'CA'),
                    ],
                ),
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
        num_proc: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        return calculate_vs(
            chains, self.make_seq_vs(), num_proc, verbose, map_name='PK',
        )

    def calculate_str_vs(
        self,
        chains: abc.Sequence[ChainStructure],
        num_proc: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        return calculate_vs(
            chains, self.make_str_vs(), num_proc, verbose, map_name='PK',
        )


def calculate_vs(
    chains: abc.Sequence[CT],
    vs: abc.Sequence[VT],
    num_proc: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    manager = Manager(verbose=verbose)
    calculator = GenericCalculator(num_proc=num_proc)
    results = manager.calculate(chains, vs, calculator, **kwargs)
    df = manager.aggregate_from_it(results, replace_errors=False)
    assert isinstance(df, pd.DataFrame), 'Failed to convert results into a table'
    return df


if __name__ == '__main__':
    raise RuntimeError
