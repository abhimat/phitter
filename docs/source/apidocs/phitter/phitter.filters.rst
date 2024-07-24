:py:mod:`phitter.filters`
=========================

.. py:module:: phitter.filters

.. autodoc2-docstring:: phitter.filters
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`filter <phitter.filters.filter>`
     - .. autodoc2-docstring:: phitter.filters.filter
          :summary:
   * - :py:obj:`naco_ks_filt <phitter.filters.naco_ks_filt>`
     - .. autodoc2-docstring:: phitter.filters.naco_ks_filt
          :summary:
   * - :py:obj:`nirc2_lp_filt <phitter.filters.nirc2_lp_filt>`
     - .. autodoc2-docstring:: phitter.filters.nirc2_lp_filt
          :summary:
   * - :py:obj:`nirc2_kp_filt <phitter.filters.nirc2_kp_filt>`
     - .. autodoc2-docstring:: phitter.filters.nirc2_kp_filt
          :summary:
   * - :py:obj:`nirc2_h_filt <phitter.filters.nirc2_h_filt>`
     - .. autodoc2-docstring:: phitter.filters.nirc2_h_filt
          :summary:
   * - :py:obj:`hst_f127m_filt <phitter.filters.hst_f127m_filt>`
     - .. autodoc2-docstring:: phitter.filters.hst_f127m_filt
          :summary:
   * - :py:obj:`hst_f139m_filt <phitter.filters.hst_f139m_filt>`
     - .. autodoc2-docstring:: phitter.filters.hst_f139m_filt
          :summary:
   * - :py:obj:`hst_f153m_filt <phitter.filters.hst_f153m_filt>`
     - .. autodoc2-docstring:: phitter.filters.hst_f153m_filt
          :summary:
   * - :py:obj:`hst_f105w_filt <phitter.filters.hst_f105w_filt>`
     - .. autodoc2-docstring:: phitter.filters.hst_f105w_filt
          :summary:
   * - :py:obj:`jwst_115w_filt <phitter.filters.jwst_115w_filt>`
     - .. autodoc2-docstring:: phitter.filters.jwst_115w_filt
          :summary:
   * - :py:obj:`jwst_212n_filt <phitter.filters.jwst_212n_filt>`
     - .. autodoc2-docstring:: phitter.filters.jwst_212n_filt
          :summary:
   * - :py:obj:`jwst_323n_filt <phitter.filters.jwst_323n_filt>`
     - .. autodoc2-docstring:: phitter.filters.jwst_323n_filt
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`lambda_Ks <phitter.filters.lambda_Ks>`
     - .. autodoc2-docstring:: phitter.filters.lambda_Ks
          :summary:

API
~~~

.. py:data:: lambda_Ks
   :canonical: phitter.filters.lambda_Ks
   :value: None

   .. autodoc2-docstring:: phitter.filters.lambda_Ks

.. py:class:: filter()
   :canonical: phitter.filters.filter

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.filters.filter

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.filter.__init__

   .. py:method:: calc_isoc_filt_ext(isoc_Ks_ext, ext_alpha)
      :canonical: phitter.filters.filter.calc_isoc_filt_ext

      .. autodoc2-docstring:: phitter.filters.filter.calc_isoc_filt_ext

   .. py:method:: __eq__(other)
      :canonical: phitter.filters.filter.__eq__

   .. py:method:: __hash__()
      :canonical: phitter.filters.filter.__hash__

   .. py:method:: __lt__(other)
      :canonical: phitter.filters.filter.__lt__

   .. py:method:: __gt__(other)
      :canonical: phitter.filters.filter.__gt__

.. py:class:: naco_ks_filt()
   :canonical: phitter.filters.naco_ks_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.naco_ks_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.naco_ks_filt.__init__

.. py:class:: nirc2_lp_filt()
   :canonical: phitter.filters.nirc2_lp_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.nirc2_lp_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.nirc2_lp_filt.__init__

   .. py:method:: calc_isoc_filt_ext(isoc_Ks_ext, ext_alpha)
      :canonical: phitter.filters.nirc2_lp_filt.calc_isoc_filt_ext

      .. autodoc2-docstring:: phitter.filters.nirc2_lp_filt.calc_isoc_filt_ext

.. py:class:: nirc2_kp_filt()
   :canonical: phitter.filters.nirc2_kp_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.nirc2_kp_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.nirc2_kp_filt.__init__

.. py:class:: nirc2_h_filt()
   :canonical: phitter.filters.nirc2_h_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.nirc2_h_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.nirc2_h_filt.__init__

.. py:class:: hst_f127m_filt()
   :canonical: phitter.filters.hst_f127m_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.hst_f127m_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.hst_f127m_filt.__init__

.. py:class:: hst_f139m_filt()
   :canonical: phitter.filters.hst_f139m_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.hst_f139m_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.hst_f139m_filt.__init__

.. py:class:: hst_f153m_filt()
   :canonical: phitter.filters.hst_f153m_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.hst_f153m_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.hst_f153m_filt.__init__

.. py:class:: hst_f105w_filt()
   :canonical: phitter.filters.hst_f105w_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.hst_f105w_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.hst_f105w_filt.__init__

.. py:class:: jwst_115w_filt()
   :canonical: phitter.filters.jwst_115w_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.jwst_115w_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.jwst_115w_filt.__init__

.. py:class:: jwst_212n_filt()
   :canonical: phitter.filters.jwst_212n_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.jwst_212n_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.jwst_212n_filt.__init__

.. py:class:: jwst_323n_filt()
   :canonical: phitter.filters.jwst_323n_filt

   Bases: :py:obj:`phitter.filters.filter`

   .. autodoc2-docstring:: phitter.filters.jwst_323n_filt

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.filters.jwst_323n_filt.__init__
