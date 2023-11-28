from pathlib import Path

import pytest
import lXtractor.core.chain as lxc
from lXtractor.core import ProteinStructure

from kinactive.parsers import StructureParser, SequenceParser

TEST_STRUCTURE_PATH = Path(__file__).parent / "data" / "4hvd.mmtf.gz"
TEST_SEQUENCE_PATH = Path(__file__).parent / "data" / "P12931.fasta"


@pytest.fixture(scope="module")
def sequence_path():
    return TEST_SEQUENCE_PATH


@pytest.fixture(scope="module")
def structure_path():
    return TEST_STRUCTURE_PATH


@pytest.fixture(scope="module")
def sequence_id(sequence_path):
    return sequence_path.name.split(".")[0]


@pytest.fixture(scope="module")
def structure_id(structure_path):
    return structure_path.name.split(".")[0]


@pytest.fixture(scope="module")
def protein_pk_structure(structure_path):
    return ProteinStructure.read(structure_path)


@pytest.fixture(scope="module")
def protein_pk_chain(protein_pk_structure):
    return lxc.ChainStructure(protein_pk_structure)


@pytest.mark.parametrize(
    "inp",
    [
        "protein_pk_structure",
        "protein_pk_chain",
        "structure_path",
        "structure_id",
        "sequence_id",
    ],
)
def test_structure_parser(inp, request):
    alphafold = "sequence" in inp
    parser = StructureParser()
    val = request.getfixturevalue(inp)
    res = parser(val, chain_ids=["A"], alphafold=alphafold)
    assert isinstance(res, list)
    assert all(isinstance(x, lxc.ChainStructure) for x in res)


@pytest.mark.parametrize("inp", ["sequence_path", "sequence_id"])
def test_sequence_parser(inp, request):
    parser = SequenceParser()
    val = request.getfixturevalue(inp)
    res = parser(val)
    assert isinstance(res, lxc.ChainSequence)
