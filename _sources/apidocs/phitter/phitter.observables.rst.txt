:py:mod:`phitter.observables`
=============================

.. py:module:: phitter.observables

.. autodoc2-docstring:: phitter.observables
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`observables <phitter.observables.observables>`
     - .. autodoc2-docstring:: phitter.observables.observables
          :summary:

API
~~~

.. py:class:: observables(obs_times=None, obs=None, obs_uncs=None, obs_filts=None, obs_types=None)
   :canonical: phitter.observables.observables

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.observables.observables

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.observables.observables.__init__

   .. py:attribute:: obs_times
      :canonical: phitter.observables.observables.obs_times
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.observables.observables.obs_times

   .. py:attribute:: obs
      :canonical: phitter.observables.observables.obs
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.observables.observables.obs

   .. py:attribute:: obs_uncs
      :canonical: phitter.observables.observables.obs_uncs
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.observables.observables.obs_uncs

   .. py:attribute:: obs_filts
      :canonical: phitter.observables.observables.obs_filts
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.observables.observables.obs_filts

   .. py:attribute:: obs_types
      :canonical: phitter.observables.observables.obs_types
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.observables.observables.obs_types

   .. py:attribute:: num_obs
      :canonical: phitter.observables.observables.num_obs
      :value: 0

      .. autodoc2-docstring:: phitter.observables.observables.num_obs

   .. py:method:: set_obs_times(obs_times)
      :canonical: phitter.observables.observables.set_obs_times

      .. autodoc2-docstring:: phitter.observables.observables.set_obs_times

   .. py:method:: set_obs(obs, obs_uncs=None)
      :canonical: phitter.observables.observables.set_obs

      .. autodoc2-docstring:: phitter.observables.observables.set_obs

   .. py:method:: set_obs_filts(obs_filts)
      :canonical: phitter.observables.observables.set_obs_filts

      .. autodoc2-docstring:: phitter.observables.observables.set_obs_filts

   .. py:method:: set_obs_types(obs_types)
      :canonical: phitter.observables.observables.set_obs_types

      .. autodoc2-docstring:: phitter.observables.observables.set_obs_types

   .. py:method:: _make_filt_search_filters()
      :canonical: phitter.observables.observables._make_filt_search_filters

      .. autodoc2-docstring:: phitter.observables.observables._make_filt_search_filters
