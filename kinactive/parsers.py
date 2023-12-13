import logging
from collections import abc
from io import StringIO
from itertools import chain
from pathlib import Path

import lXtractor.chain as lxc
from lXtractor.chain import recover
from lXtractor.core import ProteinStructure, GenericStructure
from lXtractor.ext import AlphaFold, PDB, fetch_uniprot
from lXtractor.util import read_fasta

LOGGER = logging.getLogger(__name__)


class SequenceParser:
    """
    Parser for a single sequence. Accepts a path to a ChainSequence, a UniProt
    ID, or a path to a fasta file. If the latter is provided, will read only
    the first sequence.

    >>> parser = SequenceParser()
    >>> seq = parser('P12931')
    >>> seq.id == 'P12931|1-536'
    True

    """

    @staticmethod
    def from_uniprot(query_id: str) -> lxc.ChainSequence:
        """
        Fetch a single sequence from UniProt.

        :param query_id: UniProt accession.
        :return: Initialized ChainSequence.
        """
        fetched = fetch_uniprot([query_id])
        seq_id, raw_seq = next(read_fasta(StringIO(fetched)))
        acc = seq_id.split("|")[1]
        cs = lxc.ChainSequence.from_tuple((acc, raw_seq))
        cs.meta["UniProtID"] = seq_id
        return cs

    def __call__(self, inp: str | Path | lxc.ChainSequence) -> lxc.ChainSequence:
        if isinstance(inp, lxc.ChainSequence):
            return inp
        if isinstance(inp, str):
            if Path(inp).exists():
                inp = Path(inp)
            else:
                return self.from_uniprot(inp)
        if isinstance(inp, Path):
            if inp.is_file():
                return lxc.ChainSequence.from_file(inp)
            elif inp.is_dir():
                return recover(lxc.ChainSequence.read(inp))
            else:
                raise ValueError(f"Invalid Path input {inp}")
        raise TypeError(f"Unsupported input type {type(inp)} for inp {inp}")


class StructureParser:
    """
    A parser of a single structure, and immediately split it into chains.
    Accepts a path to a file in supported format ("pdb", "cif", or "mmtf";
    can be an archive with "gz" suffix).

    >>> parser = StructureParser()
    >>> cs = parser('2oiq')
    >>> cs
    [ChainStructure(2oiq:A|1-265|), ChainStructure(2oiq:B|1-265|)]

    Accepts chain_ids to filter for.
    >>> cs = parser('2oiq', ['A'])
    >>> cs
    [ChainStructure(2oiq:A|1-265|)]

    >>> cs = parser('P12931', alphafold=True)
    >>> cs
    [ChainStructure(P12931:A|1-536|)]

    """

    def __init__(
        self, pdb_fmt: str = "mmtf.gz", af2_fmt: str = "cif", split_altloc: bool = True
    ):
        self.af2_interface = AlphaFold()
        self.pdb_interface = PDB()
        self.pdb_fmt = pdb_fmt
        self.af2_fmt = af2_fmt
        self.split_altloc = split_altloc

    def from_alphafold(self, inp: str) -> GenericStructure:
        """
        Fetch and initialize a single structure from the AlphaFold2 database.

        :param inp: Supported ID, such as UniProt accession.
        :return: A generic structure initialized.
        """
        return fetch_structure(self.af2_interface, inp, self.af2_fmt)

    def from_pdb(self, inp: str) -> GenericStructure:
        """
        Fetch and initialize a single structure from the PDB database.

        :param inp: PDB identifier.
        :return: An generic structure initialized.
        """
        return fetch_structure(self.pdb_interface, inp, self.pdb_fmt)

    def __call__(
        self,
        inp: str | Path | lxc.ChainStructure,
        chain_ids: abc.Sequence[str] | None = None,
        alphafold: bool = False,
    ) -> list[lxc.ChainStructure]:
        gs = None
        if isinstance(inp, lxc.ChainStructure):
            return [inp]
        if isinstance(inp, (GenericStructure, ProteinStructure)):
            gs = inp
        if isinstance(inp, str):
            if Path(inp).exists():
                inp = Path(inp)
            else:
                gs = self.from_alphafold(inp) if alphafold else self.from_pdb(inp)
        if isinstance(inp, Path):
            if inp.is_dir():
                return [recover(lxc.ChainStructure.read(inp))]
            elif inp.is_file():
                gs = ProteinStructure.read(inp)
            else:
                raise ValueError(f"Invalid Path input {inp}")

        if gs is None:
            raise TypeError(f"Unsupported input type {type(inp)} for inp {inp}")

        structures = gs.split_chains(polymer=True)
        if self.split_altloc:
            structures = chain.from_iterable(x.split_altloc() for x in structures)
        if chain_ids:
            structures = filter(
                lambda x: len(x.chain_ids_polymer) == 1
                and x.chain_ids_polymer[0] in chain_ids,
                structures,
            )
        return list(map(lxc.ChainStructure, structures))


def fetch_structure(
    fetcher: PDB | AlphaFold, structure_id: str, fmt: str
) -> GenericStructure | None:
    fetched, failed = fetcher.fetch_structures(
        [structure_id], dir_=None, fmt=fmt, parse=True
    )
    if failed:
        raise RuntimeError(f"Structure {structure_id} failed to fetch")
    assert len(fetched) == 1
    return fetched[0][1]


if __name__ == "__main__":
    raise RuntimeError
