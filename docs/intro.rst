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

The package is installable via ``pip``. From Pypi::

    pip install kinactive

Or directly from GitHub::

    pip install git+https://github.com/edikedik/kinactive.git

Depending on the data collection usage needs, you may need to install ``mafft``,
either directly (see the `mafft docs`_) or using ``conda``::

    conda install -c bioconda mafft

Using the data
--------------

``KinActive`` is not supplied with the raw data.
One may fetch the data accompanying the paper (see
:doc:`Fetching the data <fetching>`) or build a new raw collection (see
:doc:`Build a new lXt-PK collection <notebooks/build_raw_database>`).

Once the data is obtained, you can load the chains as:

.. code-block:: python

    from pathlib import Path
    from kinactive.db import DB
    db = DB()
    chains = db.load(Path("path/to/chains"))

.. hint::
    To speed up the loading, one may want to increase the number of cpus
    (see :class:`kinactive.config.DBConfig`).

.. hint::
    One may use an iterable over dumped ``Chain`` objects, e.g.,
    ``list(Path("path/to/chains").glob('*'))[:10]``
    and supply it to :meth:`kinactive.db.DB.load`.

This will result in a ``ChainList`` of ``Chain`` objects, each containing a
canonical UniProt ``ChainSequence`` and a list of associated ``ChainStructure``s.
See the `lXtractor docs`_ for more details on what these objects are
and how to use them.

Calculating the variables
-------------------------

Once the chains are loaded, one can use them to calculate new variables.

To calculate the default variables for loaded ``chains``.

.. code-block:: python

    from kinactive.features import DefaultFeatures
    fs = DefaultFeatures()
    # Get domains mapped to profile positions.
    domains = chains.collapse_children()
    res = fs.calculate_all_vs(domains)

.. hint::
    Provide ``base="path/to/dir"`` to automatically save the default variables

.. hint::
    Speed-up the calculation by using multiple CPUs to calculate structural
    variables via the ``num_proc`` parameters.

.. note::
    See :doc:`Calculate default variables <notebooks/calculate_default_variables>`
    for an example of variables' calculation.

Calculating non-default variables is a bit more involved and is covered in
the `lXtractor docs`_.

Using the models
----------------

To load the models, use:

.. code-block:: python

    from kinactive.io import load_dfg, load_kinactive
    ka = load_kinactive()
    dfg = load_dfg()

The first line will load the :class:`kinactive.model.KinActiveClassifier` model.
This class provides a general-purpose interface, wrapping the actual model under
the :attr:`kinactive.model.KinActiveClassifier.model` attribute. It allows to
access the :attr:`features <kinactive.model.KinActiveClassifier.model>` and
:attr:`parameters <kinactive.model.KinActiveClassifier.params>`, train, use the
model for predictions and so on.

The second line will load the :class:`kinactive.model.DFGClassifier` model.
It comprises three :class:`kinactive.model.KinActiveClassifier` objects and
a logistic regression meta-classifier outputting final predictions.

Both models can be used in the same manner. They require a dataset with
:meth:`kinactive.model.KinActiveClassifier.features` and
:meth:`kinactive.model.KinActiveClassifier.targets` columns to predict. Assuming
the ``df`` variable to encapsulate such a dataset (as a `pandas DataFrame`_).

.. code-block:: python

    ka_labels = ka.predict(df)
    dfg_labels = dfg.predict(df)

.. hint::
    :meth:`kinactive.model.DFGclassifier.predict_full` and
    :meth:`kinactive.model.KinActiveClassifier.predict_full` will preserve
    individual predictors' outputs and add columns to an initial
    `pandas DataFrame`_).

Building the distance matrix
----------------------------

The "distance matrix" is a symmetric pairwise distance matrix constructed from
the extracted domain structures. The distance is the RMSD between the DFG-Asp/
DFG-Phe of a pair of superposed domain structures. The protocol will handle
superpositions and RMSD calculations and output a new "long form" distance matrix
with four columns: ``[ID1, ID2, RMSD_CA, RMSD_DFG]``.

Assuming the ``chains`` were loaded as described in :doc:`Using the data`, i.e.,
at the level of initial ``Chain``, we'll access the structure domains and supply
them into :meth:`kinactive.distances.DistanceMatrix.build`.

.. code-block:: python

    from kinactive.distances import DistanceMatrix
    domains = chains.collapse_children().structures
    dm = DistanceMatrix().build(domains)

.. hint::
    Similar to :class:`kinactive.db.DB`, there is a config dataclass allowing
    to customize the calculation process. See :class:`kinactive.config.MatrixConfig`.

What's next?
------------

If you are interested in making a similar data collection or annotating your
PK domains, check out the :doc:`tutorial <notebooks/tutorial>`.

.. _conda docs: https://docs.anaconda.com/
.. _mafft docs: https://mafft.cbrc.jp/alignment/software/
.. _lXtractor docs: https://lxtractor.readthedocs.io/en/latest/
.. _pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
