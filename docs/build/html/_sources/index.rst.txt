.. phitter documentation master file.

Welcome to phitter's documentation!
===================================

phitter is an open-source python package to simulate and fit to data observables from stellar binary systems. At this time, observables that can be generated and fit with phitter include photometry (i.e., observed fluxes) and line-of-sight radial velocities (RVs).

Modeling of binary systems and calculation of observables is primarily handled with `PHOEBE <http://www.phoebe-project.org>`_. When computing flux from model binaries, synthetic photometry for stars is derived for a wide range of telescope and passbands using `SPISEA <https://spisea.readthedocs.io/en/latest/>`_. Parameters for the binary system's stellar components can be derived via interpolation of model stellar tracks (we currently implement `MIST <http://waps.cfa.harvard.edu/MIST/>`_). Otherwise, arbitrary stellar parameters for one or both stars can also be specified.

Fitting of observables to binary models is conducted with the use of MCMC sampling code. We provide support for sampling with nested sampling codes like `MultiNest <https://github.com/farhanferoz/MultiNest>`_ (via `PyMultiNest <https://github.com/JohannesBuchner/PyMultiNest>`_), `UltraNest <https://johannesbuchner.github.io/UltraNest/>`_, or `dynesty <https://dynesty.readthedocs.io/en/stable/index.html>`_. Example scripts to demonstrate how to set up a fitter are provided.


[Structure]

[How to set up a model binary]

[Including additional effects on photometric observables.]

[Including additional effects on RV observables.]

[How to derive own passbands and implement into PHOEBE]

[How to fit observables with nested sampling.]

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
