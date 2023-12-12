Fetching the data
=================

To manually fetch the latest `KinActive` datasets, please navigate to the
`Zenodo repository <https://zenodo.org/doi/10.5281/zenodo.10256947>`_.

The CLI implements a convenient way to download all data and separate
datasets. To view the available options, execute ``kinactive fetch``

::

    Usage: kinactive fetch [OPTIONS]

    A simple utility to fetch related data.

    All options except `out_dir` are flags.

    Usage example: `kinactive fetch --pdb_struc_pred --sp_seq_pred -vru` will
    fetch ML model predictions for SwissProt sequences and PDB structures,
    unpack fetched archives, remove them, and output basic logging info.

    To fetch the data manually, navigate to
    https://zenodo.org/doi/10.5281/zenodo.10256947

    Options:
    --db                     lXtractor data collection of PK sequences and
                           structures.
    --pdb_struc_pred         DFG and active/inactive predictions for all
                           structural domains in the lXt-PK data collection
                           produced by DFGclassifier and KinActive models.
    --pdb_struc_features     Default feature set for all structural domains in
                           the lXt-PK data collection.
    --pdb_lig_features       Default feature set of ligand variables for all
                           structural domain in the lXt-PK data collection.
    --pdb_seq_features       Default feature set for all domain structure
                           sequences in the lXt-PK data collection.
    --pdb_can_seq_features   Default feature set for domains of all canonical
                           UniProt sequences encompassed by the lXt-PK data
                           collection.
    --sp_seq_pred            Predictions of DFG-in/DFG-out conformational
                           propensities for all PK domains found in SwissProt.
    --sp_model_seq_features  A set of sequence variables necessary to run all
                           sequence-based models for all PK domains in
                           SwissProt.
    -a, --all_data           Download all available data.
    -u, --unpack             Unpack each downloaded tar.gz archive.
    -r, --rm_unpacked        Remove fetched archive after unpacking.
    -l, --links              Display data links.
    -o, --out_dir TEXT       A path to a directory where to save files. If not
                           provided, the current working directory will be
                           used.
    -v, --verbose            Output basic logging information to stdout.
    -h, --help               Show this message and exit.