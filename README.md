# KinActive

[![Documentation status](https://readthedocs.org/projects/kinactive/badge/?version=latest)](https://kinactive.readthedocs.io/en/latest/?badge=latest)
[![PyPi status](https://img.shields.io/pypi/v/kinactive.svg)](https://pypi.org/project/kinactive)
[![Python version](https://img.shields.io/pypi/pyversions/kinactive.svg)](https://pypi.org/project/kinactive)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

`KinActive` is a tool for protein kinase (PK) sequences and structures.

It's intended for two purposes:

1. Assembling, managing, and describing a collection of protein kinase sequences and
   structures.
2. Using structure- and sequence-based ML models to annotate protein kinase sequences
   and structures.

## Installation

The package can be installed via pip:

```bash
pip install kinactive
```

or directly from this repository:

```bash
pip install git+https://github.com/edikedik/kinactive.git
```

Using virtual environments (e.g., conda) is (always) recommended.

## PK data collection and models

The data and ML models were extensively described in the following papers (currently
under review):

1. _Classifying protein kinase conformations with machine learning_. Ivan Reveguk &
   Thomas Simonson.
2. _Uncovering DFG-out sequence propensity determinants of kinases with machine
   learning_. Ivan Reveguk & Thomas Simonson.

In short, the data collection encompasses PK domain PDB structures nested under (and
mapped to) the canonical UniProt sequences. Each domain sequence is also mapped to a
single reference profile (PF00069 from Pfam). The collection is prepared using the
[lXtractor](https://github.com/edikedik/lXtractor). Thus, any updates in the data are
tied to `lXtractor` updates and improvements.

All the ML models here are [XGBoost](https://xgboost.readthedocs.io/en/stable/) binary
classifiers. There are two kinds of models: those annotating PK domain sequences and
structures. The sequence-based models predict a given sequence's propensity to adopt
DFG-in or DFG-out orientations, in the apo or holo states, separately for tyrosine
and serine/threonine kinases. To facilitate the distribution into TK and STK groups,
there is an additional sequence-based model `TkST` that outputs 0 for STK domains and 1
for TKs.

The structure-based models annotate PK domain structures as active/inactive
and DFG-in/out/other. For active/inactive prediction, the model is a simple binary
classifier. On the other hand, `DFGclassifier` is a stack of three `XGBoost` models
with a `LogisticRegression` on top. The latter was trained to give more accurate
predictions for border cases, where the conformation is ambiguous. Thus, the output
will entail the probabilities of the original `XGB` models (each for in, out, and
other conformations), and the "balanced" meta-classifier probabilities, which, for
obvious cases, will not differ much from the `XGB` probabilities.

## Usage

After installing the package, the `kinactive` CLI should be available. Execute
`kinactive` in the terminal to see, which commands are available. Currently, there are
`fetch`, `db`, and `predict` commands.

### Fetching already prepared data

Use `kinactive fetch` to download already prepared datasets. One can customize, which
data to download via options. For instance, running 
```bash
kinactive fetch --db
```
will download only the PK data collection, whereas executing 
```bash
kinactive fetch --all -rvu -o downloads
```
will fetch all the available data to `./downloads`, unpack and remove raw archives, and
output basic logging information about the progress.

The datasets can also be fetched directly via 
[this link](https://zenodo.org/doi/10.5281/zenodo.10256947).

### Using ML models to predict labels

Use `kinactive predict` to apply the ML models to a small number of sequences
or structures. For a more extensive data collection, one should first compile it
separately (see below). Then variable descriptors can be calculated here or also
separately using `lXtractor` (see
[this link](https://kinactive.readthedocs.io/en/latest/notebooks/calculate_default_variables.html)
).

**Examples**:

This command will run sequence-based models on the SRC kinase sequence:

```bash
kinactive predict -t s -o ./seq_predictions P12931
```

which should output the following:

This command will run structure-based models on two SRC kinase structures:

```bash
kinactive predict -t S -o ./pdb_predictions 2OIQ 2SRC
```

Finally, the following command will run structure-based models on the
AlphaFold2-predicted model of the SRC kinase:

```bash
kinactive predict -t a -o ./af2_predictions P12931
```

In each of these cases, the collection of chain sequences or chains structures,
the calculated variables necessary for the models, and predictions will be saved
to the `*_predictions` directory. Note that these chains can also serve as input
to the models. For instance, to run sequence-based models on domain sequences
extracted from the 2OIQ and 2SRC entries, one could execute:

```bash
kinactive predict -t s -o ./str_seq_predictions -d ./pdb_predictions/chains/*/segments/*
```

Note that in the command above, we supplied a flag `-d` to signify that the domains we
provided paths to already extracted domains (stored `ChainSequence` objects). Also note
that flags can be concatenated, e.g., `-dlv` would translate to "domains were
extracted; write a log file; verbose".

### Compiling a new PK data collection

**TBD**: Using the CLI to compile the data collection is not implemented at the moment.
One can refer to
[this link](https://kinactive.readthedocs.io/en/latest/notebooks/build_raw_database.html)
for compiling the database from Python interpreter. Compiling arbitrary data collections
will be handled by a general-purpose customizable database protocol, which will be made
available with `lXtractor`>=0.2. Stay tuned!

## Advanced usage

Advanced users may compile and explore data collection, calculate additional variables, 
run ML models, from within the Python interpreter. These use-cases are described in the
[kinactive documentation](https://kinactive.readthedocs.io/en/latest/index.html).