:py:mod:`phitter.params.star_params`
====================================

.. py:module:: phitter.params.star_params

.. autodoc2-docstring:: phitter.params.star_params
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`star_params <phitter.params.star_params.star_params>`
     - .. autodoc2-docstring:: phitter.params.star_params.star_params
          :summary:
   * - :py:obj:`stellar_params_obj <phitter.params.star_params.stellar_params_obj>`
     - .. autodoc2-docstring:: phitter.params.star_params.stellar_params_obj
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ks_filt_info <phitter.params.star_params.ks_filt_info>`
     - .. autodoc2-docstring:: phitter.params.star_params.ks_filt_info
          :summary:
   * - :py:obj:`v_filt_info <phitter.params.star_params.v_filt_info>`
     - .. autodoc2-docstring:: phitter.params.star_params.v_filt_info
          :summary:
   * - :py:obj:`flux_ref_Ks <phitter.params.star_params.flux_ref_Ks>`
     - .. autodoc2-docstring:: phitter.params.star_params.flux_ref_Ks
          :summary:
   * - :py:obj:`flux_ref_V <phitter.params.star_params.flux_ref_V>`
     - .. autodoc2-docstring:: phitter.params.star_params.flux_ref_V
          :summary:
   * - :py:obj:`kp_filt <phitter.params.star_params.kp_filt>`
     - .. autodoc2-docstring:: phitter.params.star_params.kp_filt
          :summary:
   * - :py:obj:`h_filt <phitter.params.star_params.h_filt>`
     - .. autodoc2-docstring:: phitter.params.star_params.h_filt
          :summary:
   * - :py:obj:`red_law_options <phitter.params.star_params.red_law_options>`
     - .. autodoc2-docstring:: phitter.params.star_params.red_law_options
          :summary:

API
~~~

.. py:class:: star_params()
   :canonical: phitter.params.star_params.star_params

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.params.star_params.star_params

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.params.star_params.star_params.__init__

   .. py:attribute:: mass_init
      :canonical: phitter.params.star_params.star_params.mass_init
      :value: None

      .. autodoc2-docstring:: phitter.params.star_params.star_params.mass_init

   .. py:attribute:: mass
      :canonical: phitter.params.star_params.star_params.mass
      :value: None

      .. autodoc2-docstring:: phitter.params.star_params.star_params.mass

   .. py:attribute:: rad
      :canonical: phitter.params.star_params.star_params.rad
      :value: None

      .. autodoc2-docstring:: phitter.params.star_params.star_params.rad

   .. py:attribute:: lum
      :canonical: phitter.params.star_params.star_params.lum
      :value: None

      .. autodoc2-docstring:: phitter.params.star_params.star_params.lum

   .. py:attribute:: teff
      :canonical: phitter.params.star_params.star_params.teff
      :value: None

      .. autodoc2-docstring:: phitter.params.star_params.star_params.teff

   .. py:attribute:: logg
      :canonical: phitter.params.star_params.star_params.logg
      :value: 0.0

      .. autodoc2-docstring:: phitter.params.star_params.star_params.logg

   .. py:attribute:: syncpar
      :canonical: phitter.params.star_params.star_params.syncpar
      :value: 1.0

      .. autodoc2-docstring:: phitter.params.star_params.star_params.syncpar

   .. py:attribute:: filts
      :canonical: phitter.params.star_params.star_params.filts
      :value: []

      .. autodoc2-docstring:: phitter.params.star_params.star_params.filts

   .. py:attribute:: mags
      :canonical: phitter.params.star_params.star_params.mags
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.params.star_params.star_params.mags

   .. py:attribute:: mags_abs
      :canonical: phitter.params.star_params.star_params.mags_abs
      :value: 'array(...)'

      .. autodoc2-docstring:: phitter.params.star_params.star_params.mags_abs

   .. py:attribute:: pblums
      :canonical: phitter.params.star_params.star_params.pblums
      :value: None

      .. autodoc2-docstring:: phitter.params.star_params.star_params.pblums

   .. py:method:: __str__()
      :canonical: phitter.params.star_params.star_params.__str__

      .. autodoc2-docstring:: phitter.params.star_params.star_params.__str__

.. py:data:: ks_filt_info
   :canonical: phitter.params.star_params.ks_filt_info
   :value: 'get_filter_info(...)'

   .. autodoc2-docstring:: phitter.params.star_params.ks_filt_info

.. py:data:: v_filt_info
   :canonical: phitter.params.star_params.v_filt_info
   :value: 'get_filter_info(...)'

   .. autodoc2-docstring:: phitter.params.star_params.v_filt_info

.. py:data:: flux_ref_Ks
   :canonical: phitter.params.star_params.flux_ref_Ks
   :value: None

   .. autodoc2-docstring:: phitter.params.star_params.flux_ref_Ks

.. py:data:: flux_ref_V
   :canonical: phitter.params.star_params.flux_ref_V
   :value: None

   .. autodoc2-docstring:: phitter.params.star_params.flux_ref_V

.. py:data:: kp_filt
   :canonical: phitter.params.star_params.kp_filt
   :value: 'nirc2_kp_filt(...)'

   .. autodoc2-docstring:: phitter.params.star_params.kp_filt

.. py:data:: h_filt
   :canonical: phitter.params.star_params.h_filt
   :value: 'nirc2_h_filt(...)'

   .. autodoc2-docstring:: phitter.params.star_params.h_filt

.. py:data:: red_law_options
   :canonical: phitter.params.star_params.red_law_options
   :value: None

   .. autodoc2-docstring:: phitter.params.star_params.red_law_options

.. py:class:: stellar_params_obj(ext_Ks=2.63, dist=7971.0 * u.pc, filts_list=[kp_filt, h_filt], ext_law='NL18', *args, **kwargs)
   :canonical: phitter.params.star_params.stellar_params_obj

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.params.star_params.stellar_params_obj

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.params.star_params.stellar_params_obj.__init__

   .. py:method:: _calc_filts_info()
      :canonical: phitter.params.star_params.stellar_params_obj._calc_filts_info

      .. autodoc2-docstring:: phitter.params.star_params.stellar_params_obj._calc_filts_info

   .. py:method:: _create_spisea_filts_list()
      :canonical: phitter.params.star_params.stellar_params_obj._create_spisea_filts_list

      .. autodoc2-docstring:: phitter.params.star_params.stellar_params_obj._create_spisea_filts_list

   .. py:method:: calc_pblums(filt_absMags)
      :canonical: phitter.params.star_params.stellar_params_obj.calc_pblums

      .. autodoc2-docstring:: phitter.params.star_params.stellar_params_obj.calc_pblums
