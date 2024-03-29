"""
A :class:`DB` class for the PK data collection creation and io.
"""
import json
import logging
import operator as op
import typing as t
from collections import abc
from io import StringIO
from itertools import chain
from pathlib import Path
from random import sample

import pandas as pd
from lXtractor.chain import (
    Chain,
    ChainIO,
    ChainInitializer,
    ChainList,
    ChainSequence,
    ChainStructure,
    recover,
)
from lXtractor.core.config import DefaultConfig
from lXtractor.core.segment import resolve_overlaps
from lXtractor.ext import PDB, PyHMMer, SIFTS, fetch_uniprot, filter_by_method
from lXtractor.util import get_files, read_fasta, write_fasta
from more_itertools import ilen, consume, unzip
from toolz import curry, groupby, itemmap, keyfilter, keymap
from tqdm.auto import tqdm

from kinactive.base import TK_PROFILE_PATH, PK_PROFILE_PATH
from kinactive.config import DBConfig, DumpNames

T = t.TypeVar("T")
CT_: t.TypeAlias = Chain | ChainSequence | ChainStructure
LOGGER = logging.getLogger(__name__)

# Change primary polymer type to the expected protein
DefaultConfig["structure"]["primary_pol_type"] = "p"


# TODO: some object IDs are duplicated:
# This stems from the issue of chimeric sequences.
# However, such sequences should not pass the filtering when canonical/structure
# seqs are compared.
# ['ChainStructure(PK_1|10-260<-(5UFU:A|1-375))',
#  'ChainStructure(PK_1|136-354<-(7APJ:A|1-385))',
#  'ChainStructure(PK_1|38-323<-(3U87:A|1-334))',
#  'ChainStructure(PK_1|38-323<-(3U87:B|1-334))']
# TODO: include UniProt metadata
# TODO: an option to patch PDB sequences


def _get_remaining(names: abc.Iterable[str], dir_: Path) -> set[str]:
    existing = {x.stem for x in get_files(dir_).values()}
    return set(names) - existing


def _is_sequence_of_chain_seqs(
    s: abc.Sequence[t.Any],
) -> t.TypeGuard[abc.Sequence[ChainSequence]]:
    return all(isinstance(x, ChainSequence) for x in s)


def _stage_chain_init(
    seq: T, pdb_chains: abc.Iterable[str], pdb_dir: Path, fmt: str
) -> tuple[T, list[tuple[Path, list[str]]]]:
    id2chains = groupby(op.itemgetter(0), map(lambda x: x.split(":"), pdb_chains))
    path2chains = itemmap(
        lambda x: (pdb_dir / f"{x[0]}.{fmt}", list(map(op.itemgetter(1), x[1]))),
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


def _drop_all_na(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if df[c].isna().sum() == len(df)]
    return df.drop(columns=to_drop)


def _split_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    idx_parent = df.ParentID.isna()
    idx_str = df.Structure == True
    splits = (
        df[idx_parent & ~idx_str],
        df[idx_parent & idx_str],
        df[~idx_parent & ~idx_str],
        df[~idx_parent & idx_str],
    )
    return tuple(map(_drop_all_na, splits))


class DB:
    """
    An object encapsulating methods for building/saving/loading an lXtractor
    "database" -- a collection of :class:`Chain`'s.
    """

    def __init__(self, cfg: DBConfig = DBConfig()):
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
        if self._sifts is not None and not overwrite:
            sifts = self._sifts
        else:
            sifts = SIFTS(load_segments=False, load_id_mapping=True)

        if sifts.id_mapping is None:
            LOGGER.info("Initializing SIFTS for the first time.")
            sifts.fetch()
            sifts.parse()

        return sifts

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
            self._pk_hmm = PyHMMer(self.cfg.profile)
        return self._pk_hmm

    def _load_tk2pk(self) -> dict[int, int]:
        with self.cfg.tk2pk.open() as f:
            return keymap(int, json.load(f))

    def _fetch_seqs(self, ids: abc.Iterable[str]):
        raw_seqs = fetch_uniprot(
            ids,
            num_threads=self.cfg.uniprot_num_fetch_threads,
            chunk_size=self.cfg.uniprot_chunk_size,
            verbose=self.cfg.verbose,
        )
        parsed_seqs: abc.Iterable[tuple[str, str]] = read_fasta(StringIO(raw_seqs))
        if self.cfg.verbose:
            parsed_seqs = tqdm(parsed_seqs, desc="Saving fetched sequences")
        for header, seq in parsed_seqs:
            id_ = header.split("|")[1]
            write_fasta([(header, seq)], self.cfg.seq_dir / f"{id_}.fasta")

    def _read_seqs(self, ids: abc.Iterable[str]) -> ChainList[Chain]:
        files = keymap(lambda x: x.removesuffix(".fasta"), get_files(self.cfg.seq_dir))
        matching = set(files) & set(ids)
        paths = keyfilter(lambda x: x in matching, files).values()
        init = ChainInitializer(verbose=self.cfg.verbose)
        chains = list(init.from_iterable(paths))
        assert _is_sequence_of_chain_seqs(chains), "correct types are returned"
        return ChainList(map(Chain, chains))

    def _get_sifts_xray(self) -> list[str]:
        sifts = self._load_sifts()
        pdb = self._load_pdb()
        return filter_by_method(sifts.pdb_ids, pdb=pdb, method="X-ray")

    def obtain_sifts_seqs(
        self, uniprot_ids: abc.Sequence[str] | None = None
    ) -> ChainList[Chain]:
        sifts = self._load_sifts()

        if uniprot_ids:
            ids = list(filter(lambda x: x in uniprot_ids, sifts.uniprot_ids))
            LOGGER.info(
                f"Filtered to {len(ids)} out of {len(sifts.uniprot_ids)} initial IDs "
                f"contained in SIFTS using {len(uniprot_ids)} reference IDs."
            )
            missing = set(uniprot_ids) - set(ids)
            if missing:
                LOGGER.warning(f"{len(missing)} IDs were missing in SIFTS: {missing}")
        else:
            ids = sifts.uniprot_ids

        fetch_ids = _get_remaining(ids, self.cfg.seq_dir)
        LOGGER.info(f"{len(fetch_ids)} remaining sequences to fetch.")
        self._fetch_seqs(fetch_ids)

        # Read
        seqs = self._read_seqs(ids)
        LOGGER.info(f"Got {len(seqs)} seqs from {self.cfg.seq_dir}")

        # Filter sequences by size
        min_size, max_size = self.cfg.min_seq_size, self.cfg.max_seq_size
        seqs = seqs.filter(lambda s: min_size <= len(s.seq) <= max_size)
        LOGGER.info(f"Filtered to {len(seqs)} seqs in [{min_size}, {max_size}]")

        return seqs

    def discover_domains(self, seqs: ChainList[CT_]) -> ChainList[CT_]:
        def transfer_pk_map(cs: CT_) -> CT_:
            children = cs.children
            tk_children = children.filter(lambda x: "TK" in x.name)

            for c in tk_children:
                tk_df = c.seq.as_df()
                tk_df["PK"] = tk_df["TK"].map(tk2pk)
                c.seq["PK"] = tk_df["PK"].tolist()

            return cs

        @curry
        def get_field(seq: ChainSequence, contains: str) -> t.Any:
            key = next(filter(lambda x: contains in x, seq.meta))
            return seq.meta[key]

        def filter_domains(cs: CT_) -> CT_:
            children = cs.children.filter(
                lambda x: len(x.seq) >= self.cfg.pk_min_seq_domain_size
                and float(get_field(x.seq, "cov_hmm")) >= self.cfg.pk_min_cov_hmm
                and float(get_field(x.seq, "cov_seq")) >= self.cfg.pk_min_cov_seq
            )
            if len(children) == 0:
                cs.children = ChainList([])
                return cs
            non_overlapping = resolve_overlaps(
                children.sequences, value_fn=get_field(contains="score")
            )
            non_overlapping_ids = [s.id for s in non_overlapping]
            cs.children = children.filter(lambda x: x.seq.id in non_overlapping_ids)
            return cs

        tk2pk = self._load_tk2pk()
        tk_prof = PyHMMer(TK_PROFILE_PATH)
        pk_prof = PyHMMer(PK_PROFILE_PATH)

        LOGGER.info("Annotating domains")
        consume(
            pk_prof.annotate(
                seqs,
                min_size=50,
                min_score=self.cfg.pk_min_score,
                new_map_name="PK",
            )
        )
        consume(
            tk_prof.annotate(
                seqs,
                min_size=50,
                min_score=self.cfg.pk_min_score,
                new_map_name="TK",
            )
        )
        seqs = seqs.filter(lambda x: len(x.children) > 0)
        LOGGER.info(f"Discovered {len(seqs)} sequences with domain hits")

        tk_hits = seqs.collapse_children().filter(lambda x: "TK" in x.name)
        pk_hits = seqs.collapse_children().filter(lambda x: "PK" in x.name)
        LOGGER.info(f"Initial TK hits: {len(tk_hits)}")
        LOGGER.info(f"Initial PK hits: {len(pk_hits)}")

        LOGGER.info("Transferring PK profile maps to TK hits")
        seqs = (
            seqs.apply(transfer_pk_map)
            .apply(filter_domains)
            .filter(lambda x: len(x.children) > 0)
        )
        LOGGER.info(
            f"Filtered to {len(seqs)} sequences with at least one valid "
            f"domain with conforming to config criteria."
        )

        tk_hits = seqs.collapse_children().filter(lambda x: "TK" in x.name)
        pk_hits = seqs.collapse_children().filter(lambda x: "PK" in x.name)
        LOGGER.info(f"Final TK hits: {len(tk_hits)}")
        LOGGER.info(f"Final PK hits: {len(pk_hits)}")

        return seqs

    def build(
        self,
        uniprot_ids: abc.Collection[str] | None = None,
        pdb_chain_ids: abc.Collection[str] | None = None,
        n_domains: int = 0,
    ) -> ChainList[Chain]:
        """
        Build a new lXt-PK data collection.

        :param uniprot_ids: An optional list of UniProt IDs to restrict
            the db to.
        :param pdb_chain_ids: An optional collection of PDB chains to restrict
            the db to. Format: "{PDB_ID}:{ChainID}".
        :param n_domains: Use n random sequence domains. It is helpful for
            testing the pipeline.
        :return: A :class:`ChainList` of :class:`Chain` objects having at least
            one child PK domain with at least one PK domain structure passing
            the filtering thresholds.
        """

        def match_seq(s: ChainStructure) -> ChainStructure:
            s.seq.match("seq1", "seq1_canonical", as_fraction=True, save=True)
            return s

        def accept_domain_structure(c: Chain) -> Chain:
            c.transfer_seq_mapping(
                DefaultConfig["mapnames"]["seq1"], map_name_in_other="seq1_canonical"
            )

            match_name = "Match_seq1_seq1_canonical"
            c = c.apply_structures(match_seq).filter_structures(
                lambda s: (
                    len(s.seq) >= self.cfg.pk_min_str_domain_size
                    and s.seq.meta[match_name] >= self.cfg.pk_min_str_seq_match
                )
            )
            return c

        def filter_structures(c: Chain) -> Chain:
            parent_ids = [x.parent.id for x in c.children.structures]
            c.structures = c.structures.filter(lambda x: x.id in parent_ids)
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
        seqs = self.obtain_sifts_seqs(uniprot_ids)

        # Annotate PK domains and filter seqs to the annotated ones
        seqs = self.discover_domains(seqs)

        if n_domains:
            seqs = ChainList(sample(seqs, n_domains))
            LOGGER.info(f"Sampled to {len(seqs)} random initial domains.")

        # Get UniProt IDs and corresponding PDB Chains
        uni2seq = {s.id.split("|")[1]: s for s in seqs}
        uni_ids = list(uni2seq)
        pdb_chains = [x for x in map(sifts.map_id, uni_ids) if x is not None]

        # Filter PDB IDs to provided list
        if pdb_chain_ids:
            num_init = ilen(chain.from_iterable(pdb_chains))
            pdb_chains = [
                list(filter(lambda x: x in pdb_chain_ids, chain_group))
                for chain_group in pdb_chains
            ]
            LOGGER.info(
                f"Filtered to {ilen(chain.from_iterable(pdb_chains))} out "
                f"of {num_init} initially mapped PDB chains "
                f"({len(pdb_chain_ids)} reference IDs were provided for filtering)."
            )
            filtered_pairs = list(
                filter(lambda x: len(x[1]) > 0, zip(uni_ids, pdb_chains, strict=True))
            )
            LOGGER.info(
                f"Filtered to {len(filtered_pairs)} sequences mapped to at least one "
                f"PDB chain."
            )
            if not filtered_pairs:
                LOGGER.warning("All sequences were filtered out. Terminating...")
                return ChainList([])
            uni_ids, pdb_chains = map(list, unzip(filtered_pairs))

        # Filter PDB IDs to X-ray structures
        pdb_ids = {x.split(":")[0] for x in chain.from_iterable(pdb_chains)}
        LOGGER.info(f"Fetching info for {len(pdb_ids)} PDB IDs.")
        xray_pdb_ids = set(
            filter_by_method(
                pdb_ids, pdb=pdb, dir_=self.cfg.pdb_dir_info, method="X-ray"
            )
        )
        LOGGER.info(
            f"Filtered to {len(xray_pdb_ids)} X-ray PDB IDs out of {len(pdb_ids)}."
        )
        pdb_chains = [
            [c for c in cs if c.split(":")[0] in xray_pdb_ids] for cs in pdb_chains
        ]

        # Fetch X-ray structures
        LOGGER.info(f"Fetching {len(xray_pdb_ids)} X-ray structures")
        pdb.fetch_structures(xray_pdb_ids, dir_=self.cfg.pdb_dir, fmt=self.cfg.pdb_fmt)

        # Init Chain objects
        seq2pdb = dict(
            _stage_chain_init(
                uni2seq[seq_id], str_ids, self.cfg.pdb_dir, self.cfg.pdb_fmt
            )
            for seq_id, str_ids in zip(uni_ids, pdb_chains)
            if len(str_ids) > 0
        )
        init = ChainInitializer(
            tolerate_failures=self.cfg.init_tolerate_failures, verbose=self.cfg.verbose
        )
        chains: ChainList[Chain] = ChainList(
            init.from_mapping(
                seq2pdb,
                val_callbacks=[_rm_solvent, curry(_filter_by_size)(cfg=self.cfg)],
                num_proc_read_str=self.cfg.init_cpus,
                num_proc_map_numbering=self.cfg.init_map_numbering_cpus,
                num_proc_add_structure=self.cfg.init_add_structure_cpus,
                add_to_children=True,
            )
        ).filter(lambda c: len(c.structures) > 0)
        LOGGER.info(f"Initialized {len(chains)} `Chain` objects.")

        num_init = len(chains.collapse_children().structures)
        chains = chains.apply(lambda c: c.apply_children(accept_domain_structure))
        chains = chains.apply(filter_structures)
        num_curr = len(chains.collapse_children().structures)
        LOGGER.info(
            f"Filtered to {num_curr} out of {num_init} domain structures "
            f"having >={self.cfg.pk_min_str_domain_size} extracted domain size "
            f"and >={self.cfg.pk_min_str_seq_match} canonical seq match fraction."
        )

        num_init = len(chains.collapse_children())
        chains = chains.apply(
            lambda c: c.filter_children(lambda x: len(x.structures) > 0)
        )
        num_curr = len(chains.collapse_children())
        LOGGER.info(
            f"Filtered to {num_curr} out of {num_init} domains with "
            "at least one valid structure."
        )

        num_init = len(chains)
        chains = chains.filter(lambda c: len(c.children) > 0)
        LOGGER.info(
            f"Filtered to {len(chains)} chains out of {num_init} "
            "with at least one extracted domains."
        )

        for c in chains.collapse_children():
            c.transfer_seq_mapping(self.cfg.pk_map_name)
            # c.seq.children = ChainList([])

        self._chains = chains

        return chains

    def save(
        self,
        dest: Path | None = None,
        chains: abc.Iterable[Chain] | None = None,
        *,
        overwrite: bool = False,
        summary: bool = True,
    ) -> None:
        """
        Save DB sequence to file system.

        :param dest: Destination path to write seqs into.
        :param chains: Manual chains input to save. If ``None``, will use
            :attr:`chains`.
        :param overwrite: Overwrite existing data in ``dest``.
        :param summary: Compose and save summaries to ``dest``.
        :return: An iterator over paths of successfully saved chains. Consume
            to trigger saving.
        """
        chains = chains or self.chains
        dest = dest or self.cfg.target_dir
        if dest.exists():
            assert dest.is_dir(), "Path to directory"
            if not overwrite:
                files = get_files(dest)
                assert len(files) == 0, "Existing dir is not empty"
        # dest.mkdir(exist_ok=True, parents=True)
        io = ChainIO(
            num_proc=self.cfg.io_cpus, verbose=self.cfg.verbose, tolerate_failures=False
        )
        consume(io.write(chains, base=dest, str_fmt=self.cfg.pdb_fmt))
        if summary:
            summary = self.chains.summary(children=True, structures=True)
            for df, name in zip(_split_summary(summary), DumpNames.summary_file_names):
                df.to_csv(dest / name, index=False)
                LOGGER.info(f"Saved summary file {name} to {dest}")

    @staticmethod
    def _construct_paths(
        paths: abc.Iterable[Path],
        domains: bool,
        structures: bool,
    ):
        if domains:
            paths = chain.from_iterable(p.glob("segments/*") for p in paths)
        if structures:
            paths = chain.from_iterable(p.glob("structures/*") for p in paths)
        return paths

    def load(
        self,
        dump: Path | abc.Iterable[Path],
        domains: bool = True,
        sequences: bool = False,
        structures: bool = False,
        structures_sequences: bool = False,
    ) -> ChainList[Chain] | ChainList[ChainStructure] | ChainList[ChainSequence]:
        """
        Load prepared db.

        :param dump: Path with dumped :class:`Chain`s.
        :param domains: Load domains without loading parent chains.
        :param sequences: Load only canonical sequences.
        :param structures: Load structures without loading canonical sequences.
        :param structures_sequences: Load structure sequences without loading
            structures.
        :return: A chain list with initialized :class:`Chain`s.
        """

        if isinstance(dump, Path):
            dump = dump.glob("*")

        dump = list(
            self._construct_paths(dump, domains, structures or structures_sequences)
        )
        LOGGER.info(f"Got {len(dump)} initial paths to read")

        io = ChainIO(self.cfg.io_cpus, self.cfg.verbose)
        if structures:
            loader = io.read_chain_str
        elif sequences or structures_sequences:
            loader = io.read_chain_seq
        else:
            loader = io.read_chain

        chains = ChainList(
            loader(dump, callbacks=[recover], search_children=not domains)
        )
        # io = ChainIO(
        #     num_proc=self.cfg.io_cpus,
        #     verbose=self.cfg.verbose,
        #     tolerate_failures=True,
        # )
        # chain_read_it = io.read_chain(
        #     dump, callbacks=[chain_tree.recover], search_children=True
        # )
        #
        # chains = ChainList(chain_read_it)
        #
        # chains = chains.apply(
        #     chain_tree.recover,
        #     verbose=self.cfg.verbose,
        #     desc="Recovering ancestry for sequences and structures",
        # )

        # chains = read_chains(
        #     dump,
        #     children=True,
        #     seq_cfg=ChainIOConfig(verbose=self.cfg.verbose),
        #     str_cfg=ChainIOConfig(verbose=self.cfg.verbose, num_proc=self.cfg.io_cpus),
        # )
        # chains = chains.apply(
        #     chain_tree.recover,
        #     verbose=self.cfg.verbose,
        #     desc="Recovering ancestry for sequences and structures",
        # )
        if len(chains) > 0:
            LOGGER.info(f"Parsed {len(chains)} `Chain`s")
            self._chains = chains
        else:
            LOGGER.warning(f"Found no `Chain`s in {dump}")

        self._chains = chains

        return chains


if __name__ == "__main__":
    raise RuntimeError
