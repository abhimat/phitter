:py:mod:`phitter.fit.likelihood`
================================

.. py:module:: phitter.fit.likelihood

.. autodoc2-docstring:: phitter.fit.likelihood
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`log_likelihood_chisq <phitter.fit.likelihood.log_likelihood_chisq>`
     - .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq
          :summary:
   * - :py:obj:`log_likelihood_chisq_weighted_obs_type <phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type>`
     - .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type
          :summary:
   * - :py:obj:`log_likelihood_chisq_weighted_filts_and_obs_type <phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type>`
     - .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type
          :summary:

API
~~~

.. py:class:: log_likelihood_chisq(observations, *args, **kwargs)
   :canonical: phitter.fit.likelihood.log_likelihood_chisq

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq.__init__

   .. py:method:: evaluate(model_observables)
      :canonical: phitter.fit.likelihood.log_likelihood_chisq.evaluate

      .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq.evaluate

.. py:class:: log_likelihood_chisq_weighted_obs_type(observations, *args, **kwargs)
   :canonical: phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type

   Bases: :py:obj:`phitter.fit.likelihood.log_likelihood_chisq`

   .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type.__init__

   .. py:method:: evaluate(model_observables)
      :canonical: phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type.evaluate

      .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_obs_type.evaluate

.. py:class:: log_likelihood_chisq_weighted_filts_and_obs_type(observations, *args, **kwargs)
   :canonical: phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type

   Bases: :py:obj:`phitter.fit.likelihood.log_likelihood_chisq`

   .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type.__init__

   .. py:method:: evaluate(model_observables)
      :canonical: phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type.evaluate

      .. autodoc2-docstring:: phitter.fit.likelihood.log_likelihood_chisq_weighted_filts_and_obs_type.evaluate
