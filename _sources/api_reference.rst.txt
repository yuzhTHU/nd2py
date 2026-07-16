API Reference
=============

This page collects the public nd2py API in one searchable document. Use the
right-hand table of contents, the global search box, or your browser's page
search to find an object or option.

Symbolic engine
---------------

Symbol user API
^^^^^^^^^^^^^^^

The following methods are available on every concrete Symbol, including
``Variable``, ``Number``, operators, and complete expressions.

.. autoclass:: nd2py.core.symbols.symbol.Symbol
   :members: eval, eval_eic, eval_torch, to_str, to_tree, simplify, split_by_add, split_by_mul, fix_nettype, copy
   :inherited-members:
   :show-inheritance:

.. include:: api/nd2py.core.symbols.rst
   :start-line: 3

.. include:: api/nd2py.core.basic.rst
   :start-line: 3

.. include:: api/nd2py.core.calc.rst
   :start-line: 3

.. include:: api/nd2py.core.transform.rst
   :start-line: 3

.. include:: api/nd2py.core.tree.rst
   :start-line: 3

.. include:: api/nd2py.core.nettype.rst
   :start-line: 3

.. include:: api/nd2py.core.converter.rst
   :start-line: 3

.. include:: api/nd2py.core.context.rst
   :start-line: 3

Search algorithms
-----------------

.. include:: api/nd2py.search.gp.rst
   :start-line: 3

.. include:: api/nd2py.search.llmsr.rst
   :start-line: 3

.. include:: api/nd2py.search.mcts.rst
   :start-line: 3

.. include:: api/nd2py.search.ndformer.rst
   :start-line: 3

Data generation and utilities
-----------------------------

.. include:: api/nd2py.dataset.rst
   :start-line: 3

.. include:: api/nd2py.generator.rst
   :start-line: 3

.. include:: api/nd2py.utils.rst
   :start-line: 3
