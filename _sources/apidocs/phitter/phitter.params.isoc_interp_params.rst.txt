:py:mod:`phitter.params.isoc_interp_params`
===========================================

.. py:module:: phitter.params.isoc_interp_params

.. autodoc2-docstring:: phitter.params.isoc_interp_params
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`isoc_mist_stellar_params <phitter.params.isoc_interp_params.isoc_mist_stellar_params>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`lambda_Ks <phitter.params.isoc_interp_params.lambda_Ks>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.lambda_Ks
          :summary:
   * - :py:obj:`dlambda_Ks <phitter.params.isoc_interp_params.dlambda_Ks>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.dlambda_Ks
          :summary:
   * - :py:obj:`ks_filt_info <phitter.params.isoc_interp_params.ks_filt_info>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.ks_filt_info
          :summary:
   * - :py:obj:`v_filt_info <phitter.params.isoc_interp_params.v_filt_info>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.v_filt_info
          :summary:
   * - :py:obj:`flux_ref_Ks <phitter.params.isoc_interp_params.flux_ref_Ks>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.flux_ref_Ks
          :summary:
   * - :py:obj:`flux_ref_V <phitter.params.isoc_interp_params.flux_ref_V>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.flux_ref_V
          :summary:
   * - :py:obj:`kp_filt <phitter.params.isoc_interp_params.kp_filt>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.kp_filt
          :summary:
   * - :py:obj:`h_filt <phitter.params.isoc_interp_params.h_filt>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.h_filt
          :summary:
   * - :py:obj:`atm_funcs <phitter.params.isoc_interp_params.atm_funcs>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.atm_funcs
          :summary:
   * - :py:obj:`mist_phase_dict <phitter.params.isoc_interp_params.mist_phase_dict>`
     - .. autodoc2-docstring:: phitter.params.isoc_interp_params.mist_phase_dict
          :summary:

API
~~~

.. py:data:: lambda_Ks
   :canonical: phitter.params.isoc_interp_params.lambda_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.lambda_Ks

.. py:data:: dlambda_Ks
   :canonical: phitter.params.isoc_interp_params.dlambda_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.dlambda_Ks

.. py:data:: ks_filt_info
   :canonical: phitter.params.isoc_interp_params.ks_filt_info
   :value: 'get_filter_info(...)'

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.ks_filt_info

.. py:data:: v_filt_info
   :canonical: phitter.params.isoc_interp_params.v_filt_info
   :value: 'get_filter_info(...)'

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.v_filt_info

.. py:data:: flux_ref_Ks
   :canonical: phitter.params.isoc_interp_params.flux_ref_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.flux_ref_Ks

.. py:data:: flux_ref_V
   :canonical: phitter.params.isoc_interp_params.flux_ref_V
   :value: None

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.flux_ref_V

.. py:data:: kp_filt
   :canonical: phitter.params.isoc_interp_params.kp_filt
   :value: 'nirc2_kp_filt(...)'

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.kp_filt

.. py:data:: h_filt
   :canonical: phitter.params.isoc_interp_params.h_filt
   :value: 'nirc2_h_filt(...)'

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.h_filt

.. py:data:: atm_funcs
   :canonical: phitter.params.isoc_interp_params.atm_funcs
   :value: None

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.atm_funcs

.. py:data:: mist_phase_dict
   :canonical: phitter.params.isoc_interp_params.mist_phase_dict
   :value: None

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.mist_phase_dict

.. py:class:: isoc_mist_stellar_params(age=4000000.0, met=0.0, use_atm_func='merged', phase=None, *args, **kwargs)
   :canonical: phitter.params.isoc_interp_params.isoc_mist_stellar_params

   Bases: :py:obj:`phitter.params.star_params.stellar_params_obj`

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params.__init__

   .. py:method:: interp_star_params_mass_init(mass_init)
      :canonical: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_mass_init

      .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_mass_init

   .. py:method:: interp_star_params_rad(rad)
      :canonical: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_rad

      .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_rad

   .. py:method:: interp_star_params_teff(teff)
      :canonical: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_teff

      .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_teff

   .. py:method:: interp_star_params_mass(mass)
      :canonical: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_mass

      .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params.interp_star_params_mass

   .. py:method:: _flip_isochrone()
      :canonical: phitter.params.isoc_interp_params.isoc_mist_stellar_params._flip_isochrone

      .. autodoc2-docstring:: phitter.params.isoc_interp_params.isoc_mist_stellar_params._flip_isochrone
