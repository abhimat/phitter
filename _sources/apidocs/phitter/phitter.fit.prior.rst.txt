:py:mod:`phitter.fit.prior`
===========================

.. py:module:: phitter.fit.prior

.. autodoc2-docstring:: phitter.fit.prior
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`uniform_prior <phitter.fit.prior.uniform_prior>`
     - .. autodoc2-docstring:: phitter.fit.prior.uniform_prior
          :summary:
   * - :py:obj:`gaussian_prior <phitter.fit.prior.gaussian_prior>`
     - .. autodoc2-docstring:: phitter.fit.prior.gaussian_prior
          :summary:
   * - :py:obj:`const_prior <phitter.fit.prior.const_prior>`
     - .. autodoc2-docstring:: phitter.fit.prior.const_prior
          :summary:
   * - :py:obj:`prior_collection <phitter.fit.prior.prior_collection>`
     - .. autodoc2-docstring:: phitter.fit.prior.prior_collection
          :summary:

API
~~~

.. py:class:: uniform_prior(bound_lo, bound_up)
   :canonical: phitter.fit.prior.uniform_prior

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.fit.prior.uniform_prior

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.prior.uniform_prior.__init__

   .. py:method:: __call__(cube)
      :canonical: phitter.fit.prior.uniform_prior.__call__

      .. autodoc2-docstring:: phitter.fit.prior.uniform_prior.__call__

   .. py:method:: __repr__()
      :canonical: phitter.fit.prior.uniform_prior.__repr__

.. py:class:: gaussian_prior(mean, sigma)
   :canonical: phitter.fit.prior.gaussian_prior

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.fit.prior.gaussian_prior

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.prior.gaussian_prior.__init__

   .. py:method:: __call__(cube)
      :canonical: phitter.fit.prior.gaussian_prior.__call__

      .. autodoc2-docstring:: phitter.fit.prior.gaussian_prior.__call__

   .. py:method:: __repr__()
      :canonical: phitter.fit.prior.gaussian_prior.__repr__

.. py:class:: const_prior(value)
   :canonical: phitter.fit.prior.const_prior

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.fit.prior.const_prior

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.prior.const_prior.__init__

   .. py:method:: __call__(cube)
      :canonical: phitter.fit.prior.const_prior.__call__

      .. autodoc2-docstring:: phitter.fit.prior.const_prior.__call__

   .. py:method:: __repr__()
      :canonical: phitter.fit.prior.const_prior.__repr__

.. py:class:: prior_collection(priors_list)
   :canonical: phitter.fit.prior.prior_collection

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.fit.prior.prior_collection

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.prior.prior_collection.__init__

   .. py:method:: prior_transform_multinest(cube, ndim, nparam)
      :canonical: phitter.fit.prior.prior_collection.prior_transform_multinest

      .. autodoc2-docstring:: phitter.fit.prior.prior_collection.prior_transform_multinest

   .. py:method:: prior_transform_ultranest(cube)
      :canonical: phitter.fit.prior.prior_collection.prior_transform_ultranest

      .. autodoc2-docstring:: phitter.fit.prior.prior_collection.prior_transform_ultranest

   .. py:method:: prior_transform_dynesty(u)
      :canonical: phitter.fit.prior.prior_collection.prior_transform_dynesty

      .. autodoc2-docstring:: phitter.fit.prior.prior_collection.prior_transform_dynesty
