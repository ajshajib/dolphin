Configuration Files
===================

This document provides a detailed explanation of all the allowed options in the ``config.yaml`` files for ``dolphin``, but some of them are optional, as indicated. Check out the ``io_directory_example/settings`` `folder <https://github.com/ajshajib/dolphin/tree/main/io_directory_example/settings>`_ for some example config files.

.. contents:: Table of Contents
   :local:
   :depth: 2

Top-level information
---------------------

- ``lens_name``: The name of the lens system being modeled.

  - Type: ``string``
  - Example:

    .. code-block:: yaml

       lens_name: "DESJ0408-5354"

- ``band``: List of photometric bands used for modeling.

  - Type: ``list of strings``
  - Example:

    .. code-block:: yaml

       band: ["F475X", "F600LP"]

- ``psf_supersampled_factor``: *(Optional)* Factor by which the Point Spread Function (PSF) is supersampled. Default is 1.

  - Type: ``float``
  - Example:

    .. code-block:: yaml

       psf_supersampled_factor: 3

- ``pixel_size``: *(Optional)* Pixel size for each band. If not provided, it will be inferred from the image data.

  - Type: ``list of floats``
  - Example:

    .. code-block:: yaml

       pixel_size: [0.04, 0.04]

Model Section
-------------

- ``model``: Defines the components of the lens model.

  - Suboptions:

    - ``lens``: List of lens mass profiles. Supported models include: ``EPL``, ``SIE``, ``SIS``, ``SPEP``, ``PEMD``, ``SHEAR_GAMMA_PSI``.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens: ["EPL", "SHEAR_GAMMA_PSI"]

    - ``lens_light``: List of lens light profiles. Supported models include: ``SERSIC``, ``SERSIC_ELLIPSE``, ``MGE_SET``, ``MGE_SET_ELLIPSE``. The list will be duplicated for each band.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens_light: ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"]

    - ``source_light``: List of source light profiles. Supported models include: ``SERSIC_ELLIPSE``, ``SHAPELETS``. The list will be duplicated for each band.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           source_light: ["SERSIC_ELLIPSE", "SHAPELETS"]

    - ``point_source``: *(Optional)* List of point source models. Supported models include: ``LENSED_POSITION``, ``SOURCE_POSITION``. Can be an empty list for galaxy-galaxy lenses.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           point_source: ["LENSED_POSITION"]

    - ``special``: *(Optional)* String or list of special parameter types.

      - Type: ``string`` or ``list of strings``
      - Example:

        .. code-block:: yaml

           special: ["astrometric_uncertainty"]

Satellites Section
------------------

- ``satellites``: *(Optional)* Options for modeling satellite galaxies.

  - Suboptions:

    - ``centroid_init``: Initial guesses for the centroids of satellites.

      - Type: ``list of lists of floats``
      - Example:

        .. code-block:: yaml

           centroid_init: [[1, 1], [1.5, 1.5]]

    - ``centroid_bound``: Half of the box width to constrain the centroids of satellites.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           centroid_bound: 0.5

    - ``is_elliptical``: Whether each satellite is elliptical.

      - Type: ``list of booleans``
      - Example:

        .. code-block:: yaml

           is_elliptical: [true, false]


Lens Options
------------

- ``lens_option``: Additional options for the lens model.

  - Suboptions:

    - ``centroid_init``: Initial guess for the lens centroid.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           centroid_init: [0.04, -0.04]
    
    - ``centroid_bound``: Half of the box width to constrain the deflector's centroid.

      - Type: ``float``
      - Default: ``0.5``
      - Example:

        .. code-block:: yaml

           centroid_bound: 0.5

    - ``gaussian_prior``: *(Optional)* Gaussian priors for lens parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [[gamma, 2.11, 0.03], [theta_E, 1.11, 0.13]]

    - ``fix``: *(Optional)* Fix specific parameters for the lens model.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           fix:
             0:
               gamma: 2.0

    - ``limit_mass_pa_from_light``: *(Optional)* Maximum allowed difference between the position angle of the mass and light profiles.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           limit_mass_pa_from_light: 10.0

    - ``limit_mass_q_from_light``: *(Optional)* Maximum allowed difference between the axis ratio of the mass and light profiles.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           limit_mass_q_from_light: 0.1
      

Lens Light Options
------------------

- ``lens_light_option``: *(Optional)* Additional options for the lens light model.

  - Suboptions:

    - ``fix``: Fix specific parameters for the lens light profile.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           fix: {0: {"n_sersic": 4.}}

    - ``gaussian_prior``: Gaussian priors for lens light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: 
               [[R_sersic, 0.21, 0.15]]

    - ``mge_config``: *(Optional)* Configuration for MGE_SET and MGE_SET_ELLIPSE light profiles. Can be used to set the number of Gaussian components.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           mge_config:
             0:
               n_comp: 20

Source Light Options
--------------------

- ``source_light_option``: *(Optional)* Additional options for the source light model.

  - Suboptions:

    - ``gaussian_prior``: Gaussian priors for source light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [[beta, 0.15, 0.05]]

    - ``shapelet_scale_logarithmic_prior``: Whether to apply a logarithmic prior on the shapelet scale parameter.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           shapelet_scale_logarithmic_prior: true

    - ``n_max``: Maximum number of Shapelet profiles for each band.

      - Type: ``integer`` or ``list of integers``
      - Example:

        .. code-block:: yaml

           n_max: [2, 4]

Point Source Options
--------------------

- ``point_source_option``: *(Optional)* Options for point source models.

  - Suboptions:

    - ``ra_init``: Initial guess for point source RA positions.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           ra_init: [0.1, -0.1]

    - ``dec_init``: Initial guess for point source DEC positions.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           dec_init: [0.1, -0.1]

    - ``bound``: Bound width for searching the point source centroids.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           bound: 0.2

    - ``gaussian_prior``: Gaussian priors for point source parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [[ra_image, 0.1, 0.05]]

Special Options
---------------

- ``special_option``: *(Optional)* Initialization of special parameters.

  - Suboptions:

    - ``delta_x_image``: Initial spread from point source centroid in the x-axis.

      - Type: ``array of floats corresponding to the number of point sources``
      - Example:

        .. code-block:: yaml
        
           delta_x_image: [0.0, 0.0]

    - ``delta_y_image``: Initial spread from point source centroid in the y-axis.

      - Type: ``array of floats corresponding to the number of point sources``
      - Example:

        .. code-block:: yaml
        
           delta_y_image: [0.0, 0.0]

    - ``delta_image_lower``: Lower bound in spread of point source centroid sampler.

      - Type: ``float``
      - Example:

        .. code-block:: yaml
        
           delta_image_lower: -0.004

    - ``delta_image_upper``: Upper bound in spread of point source centroid sampler.

      - Type: ``float``
      - Example:

        .. code-block:: yaml
        
           delta_image_upper: 0.004

Guess Parameters
----------------

- ``guess_params``: *(Optional)* Initial guess parameter values for component models. This is commonly used to 
  overwrite default initial configurations and center the bounds of the PSO optimization process.

  - Suboptions:

    - ``lens``: Guess parameters for lens models.
    - ``lens_light``: Guess parameters for lens light models.
    - ``source``: Guess parameters for source light models.
    - ``ps``: Guess parameters for point source models.

    - Example:

      .. code-block:: yaml

         guess_params:
           lens:
             0:
               theta_E: 1.2
               e1: 0.05
               e2: -0.05

Numeric Options
---------------

- ``numeric_option``: Numerical settings for the modeling process.

  - Suboptions:

    - ``supersampling_factor``: Supersampling factor for each band.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           numeric_option:
             supersampling_factor: [2]

Fitting Options
---------------

- ``fitting``: Settings for the fitting process.

  - Suboptions:

    - ``pso``: Whether to use Particle Swarm Optimization (PSO) for fitting.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           pso: true

    - ``pso_settings``: Settings for the PSO algorithm.

      - Suboptions:

        - ``num_particle``: Number of particles in the swarm.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_particle: 50

        - ``num_iteration``: Number of iterations for PSO.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_iteration: 50

    - ``sampling``: *(Optional)* Whether to perform sampling after optimization.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           sampling: true

    - ``sampler``: The sampler to use for sampling. Currently, only ``emcee`` is fully supported.

      - Type: ``string``
      - Example:

        .. code-block:: yaml

           sampler: emcee

    - ``sampler_settings``: Settings for the sampler.

      - Suboptions:

        - ``n_burn``: Number of burn-in steps.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               n_burn: 2

        - ``n_run``: Number of sampling steps.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               n_run: 2

        - ``walkerRatio``: Ratio of walkers to parameters.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               walkerRatio: 2
    
        - ``init_samples``: *(Optional)* Initial samples for walkers.
          
          - Type: ``list of lists of floats``

    - ``psf_iteration``: *(Optional)* Whether to perform iterative PSF fitting.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           psf_iteration: true

    - ``psf_iteration_settings``: Settings for iterative PSF fitting.

      - Suboptions:

        - ``stacking_method``: Method for stacking PSFs.

          - Type: ``string``
          - Example:

            .. code-block:: yaml

               stacking_method: "median"

        - ``num_iter``: Number of PSF iterations.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_iter: 20

        - ``psf_iter_factor``: Factor for PSF iteration.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               psf_iter_factor: 0.5

        - ``keep_psf_variance_map``: Whether to keep the PSF variance map.

          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               keep_psf_variance_map: true

        - ``psf_symmetry``: Symmetry of the PSF.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               psf_symmetry: 4

        - ``block_center_neighbour``: Block center neighbour factor.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               block_center_neighbour: 0.0

- ``fitting_kwargs_list``: *(Optional)* User-provided list of fitting sequences to bypass the automated recipes in dolphin.

  - Type: ``list``
  - Example:

    .. code-block:: yaml

       fitting_kwargs_list:
         - ['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 50}]

Lenstronomy Arbitrary Keyword Arguments
---------------------------------------

- ``kwargs_model``: *(Optional)* Pass any arbitrary arguments strictly allowed in `lenstronomy.LensModel`, `lenstronomy.LightModel` inside this section.

- ``kwargs_constraints``: *(Optional)* Pass any arbitrary constraints strictly allowed in `lenstronomy.Workflow.fitting_sequence` inside this section.

  - Example:

    .. code-block:: yaml

       kwargs_constraints:
         joint_lens_with_light: [[0, 0, ['center_x', 'center_y']]]

Mask Options
------------

- ``mask``: *(Optional)* Settings for masking regions of the image.

  - Suboptions:

    - ``provided``: Set to `true` to use custom `.npy` mask files from the `settings/masks/` directory instead of using analytical masking below.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           provided: false

    - ``centroid_offset``: Offset for the centroid of the mask.

      - Type: ``list of lists of floats``
      - Example:

        .. code-block:: yaml

           centroid_offset: [[0.0, 0.0], [0.0, 0.0]]

    - ``mask_edge_pixels``: Number of edge pixels to mask.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           mask_edge_pixels: [0, 2]

    - ``radius``: Radius of the mask for each band.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           radius: [20.0, 20.0]

    - ``a``, ``b``, ``angle``: Elliptical mask parameters for each band. Used when ``radius`` is not provided.
    
      - Type: ``list of floats``
      - Example:
      
        .. code-block:: yaml
        
           a: [10.0, 10.0]
           b: [5.0, 5.0]
           angle: [0.0, 0.0]

    - ``extra_regions``: List of circular regions to mask additionally. Format is ``[ra, dec, radius]``.

      - Type: ``list of lists of lists of floats``
      - Example:

        .. code-block:: yaml

           extra_regions:
             - [[1.0, -1.0, 0.5]]
