import logging
import operator as op
import typing as t
from collections import abc
from io import StringIO
from itertools import chain
from pathlib import Path

from lXtractor.core.chain import (
    ChainInitializer,
    ChainList,
    ChainSequence,
    ChainStructure,
    Chain,
    ChainIO,
)
from lXtractor.core.chain.tree import recover
from lXtractor.core.config import SeqNames
from lXtractor.ext.hmm import PyHMMer
from lXtractor.ext.pdb_ import PDB
from lXtractor.ext.sifts import SIFTS
from lXtractor.ext.uniprot import fetch_uniprot
from lXtractor.protocols import filter_by_method
from lXtractor.util.io import get_files
from lXtractor.util.seq import read_fasta, write_fasta
from more_itertools import ilen, take
from toolz import keymap, keyfilter, groupby, itemmap, curry
from tqdm.auto import tqdm

from kinactive.config import DBConfig

CT_: t.TypeAlias = Chain | ChainSequence | ChainStructure
LOGGER = logging.getLogger(__name__)


# TODO: some object IDs are duplicated:
# ['ChainStructure(PK_1|10-260<-(5UFU:A|1-375))',
#  'ChainStructure(PK_1|136-354<-(7APJ:A|1-385))',
#  'ChainStructure(PK_1|38-323<-(3U87:A|1-334))',
#  'ChainStructure(PK_1|38-323<-(3U87:B|1-334))']


def _get_remaining(names: abc.Iterable[str], dir_: Path) -> set[str]:
    existing = {x.stem for x in get_files(dir_).values()}
    return set(names) - existing


def _is_sequence_of_chain_seqs(
    s: abc.Sequence[t.Any],
) -> t.TypeGuard[abc.Sequence[ChainSequence]]:
    return all(isinstance(x, ChainSequence) for x in s)


def _stage_chain_init(
    seq: ChainSequence, pdb_chains: abc.Iterable[str], pdb_dir: Path, fmt: str
) -> tuple[ChainSequence, list[tuple[Path, list[str]]]]:
    id2chains = groupby(op.itemgetter(0), map(lambda x: x.split(':'), pdb_chains))
    path2chains = itemmap(
        lambda x: (pdb_dir / f'{x[0]}.{fmt}', list(map(op.itemgetter(1), x[1]))),
        id2chains,
    )
    assert all(x.exists() for x in path2chains)
    return seq, list(path2chains.items())


def _filter_by_size(
    structures: list[ChainStructure], cfg: DBConfig
) -> list[ChainStructure]:
    return [s for s in structures if len(s.seq) >= cfg.pdb_str_min_size]


def _rm_solvent(structures: list[ChainStructure]) -> list[ChainStructure]:
    return [s.rm_solvent() for s in structures]


class DB:
    """
    An object encapsulating methods for building/saving/loading an lXtractor
    "database" -- a collection of :class:`Chain`'s.
    """
    def __init__(self, cfg: DBConfig):
        self.cfg = cfg
        self._sifts: SIFTS | None = None
        self._pdb: PDB | None = None
        self._pk_hmm: PyHMMer | None = None
        self._chains: ChainList[Chain] = ChainList([])

    @property
    def chains(self) -> ChainList[Chain]:
        """
        :return: Currently stored chains.
        """
        return self._chains

    def _load_sifts(self, overwrite: bool = False) -> SIFTS:
        if self._sifts is None or overwrite:
            self._sifts = SIFTS(load_segments=False, load_id_mapping=True)
        return self._sifts

    def _load_pdb(self, overwrite: bool = False) -> PDB:
        if self._pdb is None or overwrite:
            self._pdb = PDB(
                self.cfg.max_fetch_trials,
                self.cfg.pdb_num_fetch_threads,
                self.cfg.verbose,
            )
        return self._pdb

    def _load_pk_hmm(self, overwrite: bool = False) -> PyHMMer:
        if self._pk_hmm is None or overwrite:
            self._pk_hmm = PyHMMer(self.cfg.profile, bit_cutoffs='trusted')
        return self._pk_hmm

    def _fetch_seqs(self, ids: abc.Iterable[str]):
        raw_seqs = fetch_uniprot(
            ids,
            num_threads=self.cfg.uniprot_num_fetch_threads,
            chunk_size=self.cfg.uniprot_chunk_size,
            verbose=self.cfg.verbose,
        )
        parsed_seqs: abc.Iterable[tuple[str, str]] = read_fasta(StringIO(raw_seqs))
        if self.cfg.verbose:
            parsed_seqs = tqdm(parsed_seqs, desc='Saving fetched sequences')
        for (header, seq) in parsed_seqs:
            id_ = header.split('|')[1]
            write_fasta([(header, seq)], self.cfg.seq_dir / f'{id_}.fasta')

    def _read_seqs(self, ids: abc.Iterable[str]) -> ChainList[ChainSequence]:
        files = keymap(lambda x: x.removesuffix('.fasta'), get_files(self.cfg.seq_dir))
        matching = set(files) & set(ids)
        paths = keyfilter(lambda x: x in matching, files).values()
        init = ChainInitializer(num_proc=None, verbose=self.cfg.verbose)
        chains = list(init.from_iterable(paths))
        assert _is_sequence_of_chain_seqs(chains), "correct types are returned"
        return ChainList(chains)

    def _get_sifts_xray(self) -> list[str]:
        sifts = self._load_sifts()
        pdb = self._load_pdb()
        return filter_by_method(sifts.pdb_ids, pdb=pdb, method='X-ray')

    def build(self) -> ChainList[Chain]:
        """
        Build a new kinactive database.

        :return: :class:`Chain` objects having at least one child PK domain
            with at least one PK domain structure passing filtering thresholds.
        """

        def match_seq(s: ChainStructure) -> ChainStructure:
            s.seq.match('seq1', 'seq1_canonical', as_fraction=True, save=True)
            return s

        def filter_domain_str_by_canon_seq_match(c: Chain) -> Chain:
            c.transfer_seq_mapping(SeqNames.seq1, map_name_in_other='seq1_canonical')

            match_name = 'Match_seq1_seq1_canonical'
            c = c.apply_structures(match_seq).filter_structures(
                lambda s: (
                    len(s.seq) >= self.cfg.pk_min_str_domain_size
                    and s.seq.meta[match_name] >= self.cfg.pk_min_str_seq_match
                )
            )
            return c

        # 0. Init directories
        for dir_ in [
            self.cfg.target_dir,
            self.cfg.pdb_dir,
            self.cfg.seq_dir,
            self.cfg.pdb_dir_info,
        ]:
            if dir_ is not None:
                dir_.mkdir(exist_ok=True, parents=True)

        sifts = self._load_sifts()
        pdb = self._load_pdb()

        # Fetch SIFTS UniProt seqs
        fetch_ids = _get_remaining(sifts.uniprot_ids, self.cfg.seq_dir)
        LOGGER.info(f'{len(fetch_ids)} remaining sequences to fetch')
        self._fetch_seqs(fetch_ids)

        # Read and filter fetched seqs
        seqs = self._read_seqs(sifts.uniprot_ids)
        LOGGER.info(f'Got {len(seqs)} seqs from {self.cfg.seq_dir}')
        min_size, max_size = self.cfg.min_seq_size, self.cfg.max_seq_size
        seqs = seqs.filter(lambda s: min_size <= len(s) <= max_size)
        LOGGER.info(f'Filtered to {len(seqs)} seqs in [{min_size}, {max_size}]')

        # Annotate PK domains and filter seqs to the annotated ones
        pk_hmm = self._load_pk_hmm()
        ann: abc.Iterable[ChainSequence] = pk_hmm.annotate(
            seqs,
            new_map_name=self.cfg.pk_map_name,
            min_score=self.cfg.pk_min_score,
            min_size=self.cfg.pk_min_seq_domain_size,
            min_cov_hmm=self.cfg.pk_min_cov_hmm,
            min_cov_seq=self.cfg.pk_min_cov_seq,
        )
        if self.cfg.verbose:
            ann = tqdm(ann, desc='Annotating sequence domains')
        annotated_num = sum(1 for _ in ann)
        seqs = seqs.filter(lambda x: len(x.children) > 0)
        LOGGER.info(f'Found {annotated_num} PK domains within {len(seqs)} seqs')

        seqs = ChainList(take(4, seqs))
        # seqs = ChainList(islice(seqs, 150, 200))
        # seqs = seqs.filter(lambda x: 'Q03145' in x.id)

        # Get IDs UniProt IDs and corresponding PDB Chains
        uni_ids = [x.id.split('|')[1] for x in seqs]
        pdb_chains = [x for x in map(sifts.map_id, uni_ids) if x is not None]

        # Filter PDB IDs to X-ray structures
        pdb_ids = {x.split(':')[0] for x in chain.from_iterable(pdb_chains)}
        LOGGER.info(f'Fetching info for {len(pdb_ids)} PDB IDs')
        xray_pdb_ids = set(
            filter_by_method(
                pdb_ids, pdb=pdb, dir_=self.cfg.pdb_dir_info, method='X-ray'
            )
        )
        LOGGER.info(
            f'Filtered to {len(xray_pdb_ids)} X-ray PDB IDs out of {len(pdb_ids)}'
        )
        pdb_chains = [
            [c for c in cs if c.split(':')[0] in xray_pdb_ids] for cs in pdb_chains
        ]

        # Fetch X-ray structures
        LOGGER.info(f'Fetching {len(xray_pdb_ids)} X-ray structures')
        pdb.fetch_structures(xray_pdb_ids, dir_=self.cfg.pdb_dir, fmt=self.cfg.pdb_fmt)

        # Init Chain objects
        seq2pdb = dict(
            _stage_chain_init(seq, c, self.cfg.pdb_dir, self.cfg.pdb_fmt)
            for seq, c in zip(seqs, pdb_chains)
            if len(c) > 0
        )
        init = ChainInitializer(
            self.cfg.init_cpus, tolerate_failures=True, verbose=self.cfg.verbose
        )
        chains: ChainList[Chain] = ChainList(
            init.from_mapping(
                seq2pdb,
                val_callbacks=[_rm_solvent, curry(_filter_by_size)(cfg=self.cfg)],
                num_proc_map_numbering=self.cfg.init_map_numbering_cpus,
            )
        ).filter(lambda c: len(c.structures) > 0)
        LOGGER.info(f'Initialized {len(chains)} `Chain` objects')

        _chains = (
            tqdm(chains, desc='Subsetting `Chain`s by domain boundaries')
            if self.cfg.verbose
            else chains
        )
        for c in _chains:
            for seq_child in c.seq.children:
                try:
                    child = c.spawn_child(
                        seq_child.start,
                        seq_child.end,
                        seq_child.name,
                        str_map_from=SeqNames.map_canonical,
                        tolerate_failure=True,
                    )
                    pk_name = self.cfg.pk_map_name
                    child.seq.add_seq(pk_name, seq_child[pk_name])

                except Exception as e:
                    raise RuntimeError(
                        f'Failed to init child {seq_child} for Chain {c}'
                    ) from e

        num_init = ilen(chains.collapse_children().iter_structures())
        chains = chains.apply(
            lambda c: c.apply_children(filter_domain_str_by_canon_seq_match)
        )
        num_curr = ilen(chains.collapse_children().iter_structures())
        LOGGER.info(
            f'Filtered to {num_curr} out of {num_init} domain structures '
            f'having >={self.cfg.pk_min_str_domain_size} extracted domain size '
            f'and >={self.cfg.pk_min_str_seq_match} canonical seq match fraction.'
        )

        num_init = len(chains.collapse_children())
        chains = chains.apply(
            lambda c: c.filter_children(lambda x: len(x.structures) > 0)
        )
        num_curr = len(chains.collapse_children())
        LOGGER.info(
            f'Filtered to {num_curr} out of {num_init} domains with '
            'at least one valid structures'
        )

        num_init = len(chains)
        chains = chains.filter(lambda c: len(c.children) > 0)
        LOGGER.info(
            f'Filtered to {len(chains)} chains out of {num_init} '
            'with at least one extracted domains'
        )

        for c in chains.collapse_children():
            c.transfer_seq_mapping(self.cfg.pk_map_name)

        self._chains = chains

        return chains

    def save(
        self, dest: Path, chains: abc.Iterable[Chain] | None = None
    ) -> abc.Iterator[Path]:
        """
        Save DB sequence to file system.

        :param dest: Destination path to write seqs into.
        :param chains: Manual chains input to save. If ``None``, will use
            :attr:`chains`.
        :return: An iterator over paths of successfuly saved chains. Consume
            to trigger saving.
        """
        chains = chains or self.chains
        if dest.exists():
            assert dest.is_dir(), 'Path to directory'
            files = get_files(dest)
            assert len(files) == 0, 'Existing dir is empty'
        dest.mkdir(exist_ok=True, parents=True)
        io = ChainIO(
            num_proc=self.cfg.io_cpus, verbose=self.cfg.verbose, tolerate_failures=False
        )
        yield from io.write(chains, base=dest)

    def load(self, dump: Path, n: int | None = None) -> ChainList[Chain]:
        """
        Load prepared db.

        :param dump: Path with dumped :class:`Chain`s.
        :param n: Load `n` first objects. Useful for testing.
        :return: A chain list with initialized :class:`Chain`s.
        """
        io = ChainIO(
            num_proc=self.cfg.io_cpus,
            verbose=self.cfg.verbose,
            tolerate_failures=True,
        )
        chain_read_it = io.read_chain(dump, callbacks=[recover], search_children=True)
        if n is not None:
            chain_read_it = take(n, chain_read_it)

        chains = ChainList(chain_read_it)

        if len(chains) > 0:
            LOGGER.info(f'Parsed {len(chains)} `Chain`s')
            self._chains = chains
        else:
            LOGGER.warning(f'Found no `Chain`s in {dump}')
        return chains

    def fetch(self):
        # TODO: implement fetching and unpacking
        raise NotImplementedError


if __name__ == '__main__':
    raise RuntimeError
