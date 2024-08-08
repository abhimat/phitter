:py:mod:`phitter.params.blackbody_params`
=========================================

.. py:module:: phitter.params.blackbody_params

.. autodoc2-docstring:: phitter.params.blackbody_params
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`bb_stellar_params <phitter.params.blackbody_params.bb_stellar_params>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.bb_stellar_params
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`lambda_Ks <phitter.params.blackbody_params.lambda_Ks>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.lambda_Ks
          :summary:
   * - :py:obj:`dlambda_Ks <phitter.params.blackbody_params.dlambda_Ks>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.dlambda_Ks
          :summary:
   * - :py:obj:`ks_filt_info <phitter.params.blackbody_params.ks_filt_info>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.ks_filt_info
          :summary:
   * - :py:obj:`v_filt_info <phitter.params.blackbody_params.v_filt_info>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.v_filt_info
          :summary:
   * - :py:obj:`flux_ref_Ks <phitter.params.blackbody_params.flux_ref_Ks>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.flux_ref_Ks
          :summary:
   * - :py:obj:`flux_ref_V <phitter.params.blackbody_params.flux_ref_V>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.flux_ref_V
          :summary:
   * - :py:obj:`kp_filt <phitter.params.blackbody_params.kp_filt>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.kp_filt
          :summary:
   * - :py:obj:`h_filt <phitter.params.blackbody_params.h_filt>`
     - .. autodoc2-docstring:: phitter.params.blackbody_params.h_filt
          :summary:

API
~~~

.. py:data:: lambda_Ks
   :canonical: phitter.params.blackbody_params.lambda_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.blackbody_params.lambda_Ks

.. py:data:: dlambda_Ks
   :canonical: phitter.params.blackbody_params.dlambda_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.blackbody_params.dlambda_Ks

.. py:data:: ks_filt_info
   :canonical: phitter.params.blackbody_params.ks_filt_info
   :value: 'get_filter_info(...)'

   .. autodoc2-docstring:: phitter.params.blackbody_params.ks_filt_info

.. py:data:: v_filt_info
   :canonical: phitter.params.blackbody_params.v_filt_info
   :value: 'get_filter_info(...)'

   .. autodoc2-docstring:: phitter.params.blackbody_params.v_filt_info

.. py:data:: flux_ref_Ks
   :canonical: phitter.params.blackbody_params.flux_ref_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.blackbody_params.flux_ref_Ks

.. py:data:: flux_ref_V
   :canonical: phitter.params.blackbody_params.flux_ref_V
   :value: None

   .. autodoc2-docstring:: phitter.params.blackbody_params.flux_ref_V

.. py:data:: kp_filt
   :canonical: phitter.params.blackbody_params.kp_filt
   :value: 'nirc2_kp_filt(...)'

   .. autodoc2-docstring:: phitter.params.blackbody_params.kp_filt

.. py:data:: h_filt
   :canonical: phitter.params.blackbody_params.h_filt
   :value: 'nirc2_h_filt(...)'

   .. autodoc2-docstring:: phitter.params.blackbody_params.h_filt

.. py:class:: bb_stellar_params(*args, **kwargs)
   :canonical: phitter.params.blackbody_params.bb_stellar_params

   Bases: :py:obj:`phitter.params.star_params.stellar_params_obj`

   .. autodoc2-docstring:: phitter.params.blackbody_params.bb_stellar_params

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.params.blackbody_params.bb_stellar_params.__init__

   .. py:method:: calc_star_params(mass, rad, teff)
      :canonical: phitter.params.blackbody_params.bb_stellar_params.calc_star_params

      .. autodoc2-docstring:: phitter.params.blackbody_params.bb_stellar_params.calc_star_params

   .. py:method:: get_bb_mags(bb_temp, bb_rad, diagnostic_plot=False)
      :canonical: phitter.params.blackbody_params.bb_stellar_params.get_bb_mags

      .. autodoc2-docstring:: phitter.params.blackbody_params.bb_stellar_params.get_bb_mags
