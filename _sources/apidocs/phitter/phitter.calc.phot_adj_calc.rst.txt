:py:mod:`phitter.calc.phot_adj_calc`
====================================

.. py:module:: phitter.calc.phot_adj_calc

.. autodoc2-docstring:: phitter.calc.phot_adj_calc
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`apply_distance_modulus <phitter.calc.phot_adj_calc.apply_distance_modulus>`
     - .. autodoc2-docstring:: phitter.calc.phot_adj_calc.apply_distance_modulus
          :summary:
   * - :py:obj:`apply_extinction <phitter.calc.phot_adj_calc.apply_extinction>`
     - .. autodoc2-docstring:: phitter.calc.phot_adj_calc.apply_extinction
          :summary:
   * - :py:obj:`apply_mag_shift_filt <phitter.calc.phot_adj_calc.apply_mag_shift_filt>`
     - .. autodoc2-docstring:: phitter.calc.phot_adj_calc.apply_mag_shift_filt
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`red_law_funcs_ks <phitter.calc.phot_adj_calc.red_law_funcs_ks>`
     - .. autodoc2-docstring:: phitter.calc.phot_adj_calc.red_law_funcs_ks
          :summary:

API
~~~

.. py:data:: red_law_funcs_ks
   :canonical: phitter.calc.phot_adj_calc.red_law_funcs_ks
   :value: None

   .. autodoc2-docstring:: phitter.calc.phot_adj_calc.red_law_funcs_ks

.. py:function:: apply_distance_modulus(bin_observables, target_dist)
   :canonical: phitter.calc.phot_adj_calc.apply_distance_modulus

   .. autodoc2-docstring:: phitter.calc.phot_adj_calc.apply_distance_modulus

.. py:function:: apply_extinction(bin_observables, isoc_Ks_ext, ref_filt, target_ref_filt_ext, isoc_red_law='NL18', ext_alpha=None)
   :canonical: phitter.calc.phot_adj_calc.apply_extinction

   .. autodoc2-docstring:: phitter.calc.phot_adj_calc.apply_extinction

.. py:function:: apply_mag_shift_filt(bin_observables, filt, mag_shift)
   :canonical: phitter.calc.phot_adj_calc.apply_mag_shift_filt

   .. autodoc2-docstring:: phitter.calc.phot_adj_calc.apply_mag_shift_filt
