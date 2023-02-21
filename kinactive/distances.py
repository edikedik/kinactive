from __future__ import annotations

import logging
import warnings
from collections import abc, defaultdict
from functools import reduce
from itertools import filterfalse
from pathlib import Path
from random import choice

import biotite.structure as bst
import numpy as np
import pandas as pd
from lXtractor.core.chain import ChainStructure, ChainList
from lXtractor.protocols import superpose_pairwise, SupOutputFlex, SupOutputStrict

from kinactive.config import (
    DumpNames,
    ColNames,
    DefaultMatrixConfig as DefCfg,
    MatrixConfig,
)
from kinactive.io import load_txt_lines, save_txt_lines

LOGGER = logging.getLogger(__name__)


class DistanceMatrix:
    def __init__(
        self,
        df: pd.DataFrame,
        pos_ca: list[int],
    ):
        self.df = df
        self.pos_ca = pos_ca

    @classmethod
    def build(
        cls,
        structures: abc.Iterable[ChainStructure],
        cfg: MatrixConfig = DefCfg,
    ) -> DistanceMatrix:

        structures = ChainList(structures)
        LOGGER.info(f"Initial number of structures: {len(structures)}")

        pos_map, pos = super_pos(structures, cfg.n_super_pos)
        LOGGER.info(f"Obtained {len(pos)} super positions: {pos}")

        ids = reduce(lambda x, y: x & y, map(set, (pos_map[p] for p in pos)))
        structures = structures.filter(lambda x: x.id in ids).filter_pos(
            pos + list(cfg.df_pos), map_name="PK"
        )
        LOGGER.info(
            f"Filtered to {len(structures)} with CA atoms specified by super positions "
            f"and DFG-Asp, DFG-Phe present"
        )
        results = superpose_pairwise(
            structures,
            selection_superpose=(pos, cfg.bb_atoms),
            selection_rmsd=(cfg.df_pos, [cfg.asp_atoms, cfg.phe_atoms]),
            map_name=cfg.pk_map_name,
            strict=False,
            verbose=True,
            num_proc=cfg.n_proc,
            chunksize=cfg.chunksize,
        )

        df = pd.DataFrame(
            (
                (res.ID_fix, res.ID_mob, res.RmsdSuperpose, res.RmsdTarget)
                for res in results
            ),
            columns=[
                ColNames.id_fix,
                ColNames.id_mob,
                ColNames.rmsd_ca,
                ColNames.rmsd_df,
            ],
        )
        return cls(df, pos)

    def save(self, base_path: Path = DefCfg.dir) -> None:
        self.df.to_csv(base_path / DumpNames.distances, index=False)
        save_txt_lines(self.pos_ca, base_path / DumpNames.positions_ca)

    @classmethod
    def load(cls, base_path: Path):
        df = pd.read_csv(base_path / DumpNames.distances)
        pos_ca = list(map(int, load_txt_lines(base_path / DumpNames.positions_ca)))
        return cls(df, pos_ca)

    def closest_to(
        self, id_: str, n: int, col: str = ColNames.rmsd_ca
    ) -> abc.Generator[str, None, None]:
        df = self.df
        sub = df[(df.ID1 == id_) | (df.ID2 == id_)].sort_values(col).head(n)
        for _, row in sub.iterrows():
            id1, id2 = row[ColNames.ID1], row[ColNames.ID2]
            yield id1 if id1 != id_ else id2

    def superpose(
        self,
        structures: abc.Iterable[ChainStructure],
        choose_ref_by: str = ColNames.rmsd_ca,
        **kwargs,
    ) -> list[SupOutputStrict] | list[SupOutputFlex]:
        structures = ChainList(structures)
        ids = {x.id for x in structures}
        df = self.df
        sub = df[df[ColNames.id_fix].isin(ids) & df[ColNames.id_mob].isin(ids)]
        ids_df = set(sub[ColNames.id_fix]) | set(sub[ColNames.id_mob])
        missing = ids - ids_df
        if missing:
            warnings.warn(
                f"Missing {missing} in clustering results. This may cause choosing "
                "non-optimal reference for superposition."
            )
        if missing == ids:
            ref_id = choice(list(ids))
            warnings.warn(
                "No structures are in clustering results. "
                f"Randomly picked {ref_id} for reference."
            )
        else:
            dists = {
                x: sub.loc[
                    (sub[ColNames.id_fix] == x) | (sub[ColNames.id_mob] == x),
                    choose_ref_by,
                ].mean()
                for x in ids_df
            }
            ref_id = min(dists.items(), key=lambda x: x[1])[0]

        ref = structures[ref_id].pop()
        mobile = structures.filter(lambda x: x.id != ref_id)
        assert len(mobile) + 1 == len(structures), "Total number matches"

        results = list(
            superpose_pairwise(
                [ref],
                mobile,
                selection_superpose=(self.pos_ca, DefCfg.bb_atoms),
                selection_rmsd=(DefCfg.df_pos, [DefCfg.asp_atoms, DefCfg.phe_atoms]),
                map_name=DefCfg.pk_map_name,
                **kwargs,
            )
        )
        for r, s in zip(results, mobile):
            assert s.id == r.ID_mob
            superposed = bst.superimpose_apply(s.pdb.structure.array, r.Transformation)
            s.pdb.structure = s.pdb.structure.__class__(
                superposed, s.pdb.structure.pdb_id
            )
        return results

    def matrix_ids(self, n: int) -> abc.Generator[str, None, None]:
        """
        :param n: The number of observations used for constructing the matrix.
        :return: An iterator over IDs of original observations.
        """
        df = self.df
        p1 = 0
        for k in range(n - 1, 0, -1):
            p2 = p1 + k
            chunk = set(df[ColNames.id_fix][p1: p2])
            p1 = p1 + k
            assert len(chunk) == 1
            yield chunk.pop()
        yield df[ColNames.id_mob].iloc[-1]


def ca_pos(s):
    a = s.array
    m = s.seq.get_map("numbering")
    ca_str_pos = set(a[a.atom_name == "CA"].res_id)
    return map(
        int, filterfalse(lambda x: np.isnan(x), (m.get(p, None).PK for p in ca_str_pos))
    )


def ca_pos_per_str(strs: abc.Iterable[ChainStructure]):
    d = defaultdict(list)
    for s in strs:
        for p in ca_pos(s):
            d[p].append(s.id)
    return d


def super_pos(strs: abc.Iterable[ChainStructure], n: int):
    pos_map = ca_pos_per_str(strs)
    pos_count = {k: len(v) for k, v in pos_map.items()}
    pos_count_top = sorted(pos_count.items(), key=lambda x: x[1], reverse=True)[:n]
    return pos_map, sorted(x[0] for x in pos_count_top)


if __name__ == "__main__":
    raise ValueError
