:py:mod:`phitter.legacy_lc_calc`
================================

.. py:module:: phitter.legacy_lc_calc

.. autodoc2-docstring:: phitter.legacy_lc_calc
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`phased_obs <phitter.legacy_lc_calc.phased_obs>`
     - .. autodoc2-docstring:: phitter.legacy_lc_calc.phased_obs
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`kp_filt <phitter.legacy_lc_calc.kp_filt>`
     - .. autodoc2-docstring:: phitter.legacy_lc_calc.kp_filt
          :summary:
   * - :py:obj:`h_filt <phitter.legacy_lc_calc.h_filt>`
     - .. autodoc2-docstring:: phitter.legacy_lc_calc.h_filt
          :summary:

API
~~~

.. py:data:: kp_filt
   :canonical: phitter.legacy_lc_calc.kp_filt
   :value: 'nirc2_kp_filt(...)'

   .. autodoc2-docstring:: phitter.legacy_lc_calc.kp_filt

.. py:data:: h_filt
   :canonical: phitter.legacy_lc_calc.h_filt
   :value: 'nirc2_h_filt(...)'

   .. autodoc2-docstring:: phitter.legacy_lc_calc.h_filt

.. py:function:: phased_obs(observation_times, binary_period, t0, filts_list=[kp_filt, h_filt])
   :canonical: phitter.legacy_lc_calc.phased_obs

   .. autodoc2-docstring:: phitter.legacy_lc_calc.phased_obs
