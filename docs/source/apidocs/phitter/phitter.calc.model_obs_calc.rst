:py:mod:`phitter.calc.model_obs_calc`
=====================================

.. py:module:: phitter.calc.model_obs_calc

.. autodoc2-docstring:: phitter.calc.model_obs_calc
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`binary_star_model_obs <phitter.calc.model_obs_calc.binary_star_model_obs>`
     - .. autodoc2-docstring:: phitter.calc.model_obs_calc.binary_star_model_obs
          :summary:
   * - :py:obj:`single_star_model_obs <phitter.calc.model_obs_calc.single_star_model_obs>`
     - .. autodoc2-docstring:: phitter.calc.model_obs_calc.single_star_model_obs
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`kp_filt <phitter.calc.model_obs_calc.kp_filt>`
     - .. autodoc2-docstring:: phitter.calc.model_obs_calc.kp_filt
          :summary:
   * - :py:obj:`h_filt <phitter.calc.model_obs_calc.h_filt>`
     - .. autodoc2-docstring:: phitter.calc.model_obs_calc.h_filt
          :summary:

API
~~~

.. py:data:: kp_filt
   :canonical: phitter.calc.model_obs_calc.kp_filt
   :value: 'nirc2_kp_filt(...)'

   .. autodoc2-docstring:: phitter.calc.model_obs_calc.kp_filt

.. py:data:: h_filt
   :canonical: phitter.calc.model_obs_calc.h_filt
   :value: 'nirc2_h_filt(...)'

   .. autodoc2-docstring:: phitter.calc.model_obs_calc.h_filt

.. py:class:: binary_star_model_obs(bin_observables, use_blackbody_atm=False, use_compact_object=False, print_diagnostics=False, par_compute=False, num_par_processes=8, *args, **kwargs)
   :canonical: phitter.calc.model_obs_calc.binary_star_model_obs

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.calc.model_obs_calc.binary_star_model_obs

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.calc.model_obs_calc.binary_star_model_obs.__init__

   .. py:method:: compute_obs(star1_params, star2_params, binary_params, irrad_frac_refl=1.0, num_triangles=1500, make_mesh_plots=False, mesh_plot_phases=np.array([0.25]), animate=False, mesh_plot_fig=None, mesh_plot_subplot_grid=None, mesh_plot_subplot_grid_indexes=None, mesh_temp=False, mesh_temp_cmap=None, plot_name=None, mesh_plot_kwargs={})
      :canonical: phitter.calc.model_obs_calc.binary_star_model_obs.compute_obs

      .. autodoc2-docstring:: phitter.calc.model_obs_calc.binary_star_model_obs.compute_obs

.. py:class:: single_star_model_obs(sing_star_observables, use_blackbody_atm=False, use_compact_object=False, print_diagnostics=False, par_compute=False, num_par_processes=8, *args, **kwargs)
   :canonical: phitter.calc.model_obs_calc.single_star_model_obs

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: phitter.calc.model_obs_calc.single_star_model_obs

   .. rubric:: Initialization

   .. autodoc2-docstring:: phitter.calc.model_obs_calc.single_star_model_obs.__init__

   .. py:method:: compute_obs(star1_params, num_triangles=1500, make_mesh_plots=False, mesh_plot_fig=None, mesh_plot_subplot_grid=None, mesh_plot_subplot_grid_indexes=None, mesh_temp=False, mesh_temp_cmap=None, plot_name=None, mesh_plot_kwargs={})
      :canonical: phitter.calc.model_obs_calc.single_star_model_obs.compute_obs

      .. autodoc2-docstring:: phitter.calc.model_obs_calc.single_star_model_obs.compute_obs
