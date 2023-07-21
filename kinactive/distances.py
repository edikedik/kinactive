"""
Distance matrix computation and io.
"""
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
from lXtractor.core.chain import ChainList, ChainStructure
from lXtractor.core.exceptions import MissingData
from lXtractor.protocols import SupOutputFlex, SupOutputStrict, superpose_pairwise

from kinactive.config import (
    ColNames,
    DefaultMatrixConfig as DefCfg,
    DumpNames,
    MatrixConfig,
)
from kinactive.io import load_txt_lines, save_txt_lines

LOGGER = logging.getLogger(__name__)


class DistanceMatrix:
    """
    A symmetric distance matrix encapsulating pairwise RMSD of
    superposed structures.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pos_sup: list[int],
    ):
        #: A table with three columns: (1-2) object IDs,
        #: (3) RMSD of super positions, and (4) RMSD of target positions.
        #: It is assumed to be sorted by object IDs and contain combinations
        #: ``itertools.combinations(ids, 2)`` would output.
        self.df: pd.DataFrame = df
        #: A list of positions used for superposing pairs of structures.
        #: A position is "covered" if (1) it was successfully mapped to a
        #: reference, and (2) it has a "CA" atom.
        self.pos_sup: list[int] = pos_sup

    @classmethod
    def build(
        cls,
        structures: abc.Iterable[ChainStructure],
        cfg: MatrixConfig = DefCfg,
    ) -> DistanceMatrix:
        """
        Build a new distance matrix from provided structures.

        The method will obtain a list of positions most covered by the reference.
        It will use these in a superposition protocol defined in ``lXtractor``.

        :param structures: A list of chain structures mapped to a single
            reference.
        :param cfg: A configuration file. The options are explained within
            (:class:`kinactive.config.MatrixConfig`).
        :return: The constructed distance matrix.
        """

        structures = ChainList(structures)
        LOGGER.info(f"Initial number of structures: {len(structures)}")

        pos_map, pos = super_pos(structures, cfg.n_super_pos, cfg.pk_map_name)
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
        """
        Save the distance matrix data -- a dataframe and a list of
        super-positions. The file names are hardcoded.

        :param base_path: base dir.
        """
        self.df.to_csv(base_path / DumpNames.distances, index=False)
        save_txt_lines(self.pos_sup, base_path / DumpNames.positions_ca)

    @classmethod
    def load(cls, base_path: Path):
        """
        Load the distance matrix data and initialize a new instance.

        :param base_path: base dir.
        :return: A new instance.
        """
        df = pd.read_csv(base_path / DumpNames.distances)
        pos_sup = list(map(int, load_txt_lines(base_path / DumpNames.positions_ca)))
        return cls(df, pos_sup)

    def closest_to(
        self, id_: str, n: int, col: str = ColNames.rmsd_ca
    ) -> abc.Generator[str, None, None]:
        """
        Find ``n`` structures closest to some structure.

        :param id_: An ID of a structure.
        :param n: How many closest structures to find.
        :param col:
        :return: A generator of closest IDs.
        """
        df = self.df
        sub = df[(df.ID1 == id_) | (df.ID2 == id_)].sort_values(col).head(n)
        for _, row in sub.iterrows():
            id1, id2 = row[ColNames.ID1], row[ColNames.ID2]
            yield id1 if id1 != id_ else id2

    def superpose(
        self,
        structures: abc.Iterable[ChainStructure],
        choose_ref_by: str = ColNames.rmsd_ca,
        key: str = "min",
        **kwargs,
    ) -> list[SupOutputStrict] | list[SupOutputFlex]:
        """
        Superpose a group of structures to a single reference structure.
        The latter is a structure having minimum average distance to other
        structures in the list. Consequently, the method assumes that the
        distance matrix encompasses the provided structures and will warn
        a user if it's not the case.

        :param structures: An iterable with structures to superpose.
        :param choose_ref_by: A column name in a distance matrix to choose the
            reference structure by.
        :param key: A selector of averaged distances to choose the reference by;
            either "min" or "max".
        :param kwargs: passed to :func:`superpose_pairwise` protocol.
        :return: It will return the original :func:`superpose_pairwise` output
            and transform the coordinates of the provided structures according
            to this output (inplace).
        """
        try:
            _key = {"min": min, "max": max}[key]
        except KeyError as e:
            raise ValueError("Invalid key") from e

        structures = ChainList(structures)
        if len(structures) == 1:
            raise MissingData("A single structure has nothing to superpose to")
        if len(structures) == 0:
            raise MissingData("No structures to superpose")

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
            ref_id = _key(dists.items(), key=lambda x: x[1])[0]

        ref = structures[ref_id].pop()
        mobile = structures.filter(lambda x: x.id != ref_id)
        assert len(mobile) + 1 == len(structures), "Total number matches"

        results = list(
            superpose_pairwise(
                [ref],
                mobile,
                selection_superpose=(self.pos_sup, DefCfg.bb_atoms),
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
            chunk = set(df[ColNames.id_fix][p1:p2])
            p1 = p1 + k
            assert len(chunk) == 1
            yield chunk.pop()
        yield df[ColNames.id_mob].iloc[-1]

    def fetch(self):
        raise NotImplementedError


def covered_pos(s: ChainStructure, ref_name: str) -> abc.Iterator[int]:
    """
    Get a list of covered positions. A position is "covered" if (1) it was
    successfully mapped to a reference, and (2) it has a "CA" atom.

    :param s: A chain structure.
    :param ref_name: Reference object name structure sequences were mapped to.
    :return: An iterator over covered positions.
    """
    a = s.array
    m = s.seq.get_map("numbering")
    ca_str_pos = set(a[a.atom_name == "CA"].res_id)
    mapped_pos = filter(bool, (m.get(p, None) for p in ca_str_pos))
    mapped_pos = (x._asdict()[ref_name] for x in mapped_pos)
    return map(int, filterfalse(np.isnan, mapped_pos))


def ca_pos_per_str(
    strs: abc.Iterable[ChainStructure], ref_name: str
) -> dict[int, list[str]]:
    """
    Get a mapping from HMM positions to a list of structure they covered.

    :param strs: An iterable over chain structures.
    :param ref_name: Reference object name structure sequences were mapped to.
    :return: A dictionary ``Pos => [IDS]``.
    """
    d = defaultdict(list)
    for s in strs:
        for p in covered_pos(s, ref_name):
            d[p].append(s.id)
    return d


def super_pos(
    strs: abc.Iterable[ChainStructure], n: int, ref_name: str
) -> tuple[dict[int, list[str]], list[int]]:
    """
    Get coverage data of reference positions and find the most covered positions.

    :param strs: An iterable over chain structures.
    :param n: The number of positions to get.
    :param ref_name: Reference object name structure sequences were mapped to.
    :return: A tuple with mappings ``Pos => [IDS]`` and a list of ``n`` most
        covered positions.
    """
    pos_map = ca_pos_per_str(strs, ref_name)
    pos_count = {k: len(v) for k, v in pos_map.items()}
    pos_count_top = sorted(pos_count.items(), key=lambda x: x[1], reverse=True)[:n]
    return pos_map, sorted(x[0] for x in pos_count_top)


if __name__ == "__main__":
    raise ValueError
