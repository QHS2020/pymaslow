Installation
============

From PyPI
---------

.. code-block:: bash

   pip install pymaslow

For Markov-chain graph visualization (requires ``networkx``):

.. code-block:: bash

   pip install "pymaslow[plot]"

For running the test suite:

.. code-block:: bash

   pip install "pymaslow[test]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/QHS2020/pymaslow.git
   cd pymaslow
   pip install -e .

Requirements
------------

- Python >= 3.9
- NumPy, SciPy, pandas, Matplotlib, tqdm
- (optional) networkx — graph visualization of Markov chains
- (optional) pytest — running the tests

Verifying the installation
--------------------------

.. code-block:: python

   import pymaslow
   from pymaslow import vonMisesMixture as vmmm

   print(pymaslow.__version__)
   print(vmmm.p_x)      # pre-fitted hierarchy prior, loaded at import
   print(len(pymaslow.load_compendium()))  # 823 annotated activities
