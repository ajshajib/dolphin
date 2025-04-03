Configuration File Documentation
================================

This document provides a detailed explanation of all the possible options in the `config.yaml` file for the `dolphin` pipeline. Check out the ``io_directory_example/settings`` `folder <https://github.com/ajshajib/dolphin/tree/main/io_directory_example/settings>`_ for some example config files.

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

- ``pixel_size``: *(Optional)* Pixel size for each band. If not provided, it will be inferred from the image data.

  - Type: ``list of floats``
  - Example:

    .. code-block:: yaml

       pixel_size: [0.04, 0.04]

Model Section
-------------

- ``model``: Defines the components of the lens model.

  - Suboptions:

    - ``lens``: List of lens mass profiles.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens: ["EPL", "SHEAR_GAMMA_PSI"]

    - ``lens_light``: List of lens light profiles. The list will be duplicated for each band.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens_light: ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"]

    - ``source_light``: List of source light profiles. The list will be duplicated for each band.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           source_light: ["SERSIC_ELLIPSE", "SHAPELETS"]

    - ``point_source``: *(Optional)* List of point source models. Can be an empty list for galaxy-galaxy lenses.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           point_source: ["LENSED_POSITION"]

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

    - ``constrain_position_angle_from_lens_light``: *(Optional)* Maximum allowed difference between the position angle of the mass and light profiles.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           constrain_position_angle_from_lens_light: 15

    - ``limit_mass_eccentricity_from_light``: *(Optional)* Whether to limit the mass eccentricity based on the light profile.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           limit_mass_eccentricity_from_light: true

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

    - ``n_max``: Maximum number of Sersic profiles for each band.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           n_max: [2, 4]

Numeric Options
---------------

- ``numeric_option``: Numerical settings for the modeling process.

  - Suboptions:

    - ``supersampling_factor``: Supersampling factor for each band.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

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

    - ``sampler``: The sampler to use for sampling.

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

Mask Options
------------

- ``mask``: *(Optional)* Settings for masking regions of the image.

  - Suboptions:

    - ``centroid_offset``: Offset for the centroid of the mask.

      - Type: ``list of lists of floats``
      - Example:

        .. code-block:: yaml

           centroid_offset: [[0.0, 0], [0.0, 0]]

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
