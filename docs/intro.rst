Introduction
============

``KinActive`` is a package to explore the structural kinome.
It allows to prepare/load/explore the ``lXt-PK`` data collection and use the
classifier models predicting DFG and active/inactive states of a PK conformation
from structural features.

Installation
------------

It is recommended to first create a ``conda`` virtual environment::

    conda create -n kinactive
    conda activate kinactive

See the `conda docs`_ for further details.

The package is installable via ``pip``::

    pip install kinactive

Or directly from GitHub::

    pip install git+https://github.com/edikedik/kinactive.git

Depending on the data collection usage needs, you may need to install ``mafft``,
either directly (see the `mafft docs`_) or using ``conda``::

    conda install -c bioconda mafft

Using the data
--------------

``KinActive`` is not supplied with the raw data.
One may fetch the data accompanying the paper
(see :doc:`Fetching the data <notebooks/build_raw_database>`) or build a new
raw collection (see :doc:`the notebook here <notebooks/calculate_default_variables>`).

Once the data is obtained, loading becomes trivial:

.. code-block:: Python
    from pathlib import Path
    from kinactive.db import DB
    db = DB()
    chains = db.load(Path("path/to/chains"))

.. hint::
    To speed up the loading, one may want to increase the number of cpus
    (see :class:`kinactive.config.DBConfig`).

.. hint::
    One may provide iterable over dumped ``Chain`` objects, e.g.,
    ``list(Path("path/to/chains").glob('*'))[:10]``
    to :meth:`kinactive.db.DB.load`.

This will result in a ``ChainList`` of ``Chain`` objects, each containing a
canonical UniProt sequence and a list of associated structures.
See the `lXtractor docs`_ on how to use ``Chain`` objects.

.. _conda docs: https://docs.anaconda.com/
.. _mafft docs: https://mafft.cbrc.jp/alignment/software/
.. _lXtractor docs: https://lxtractor.readthedocs.io/en/latest/

