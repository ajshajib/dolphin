Configuration File Documentation
================================

This document provides a detailed explanation of all the possible options in the `config.yaml` file for the `dolphin` pipeline. Check out the `io_directory_example/settings` folder for some example config files.

Top-Level Options
-----------------

- ``lens_name``:

  - Description: The name of the lens system being modeled.
  - Type: ``string``
  - Example:

    .. code-block:: yaml

       lens_name: "DESJ0408-5354"

- ``band``:

  - Description: List of photometric bands used for modeling.
  - Type: ``list of strings``
  - Example:

    .. code-block:: yaml

       band: ['F814W']

- ``pixel_size``:

  - Description: Pixel size for each band. If not provided, it will be inferred from the image data.
  - Type: ``list of floats``
  - Example:

    .. code-block:: yaml

       pixel_size: [0.05]

Model Section
-------------

- ``model``:

  - Description: Defines the components of the lens model.
  - Suboptions:

    - ``lens``:

      - Description: List of lens mass profiles.
      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens: ['EPL', 'SHEAR_GAMMA_PSI']

    - ``lens_light``:

      - Description: List of lens light profiles.
      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens_light: ['SERSIC_ELLIPSE']

    - ``source_light``:

      - Description: List of source light profiles.
      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           source_light: ['SERSIC_ELLIPSE']

    - ``point_source``:

      - Description: List of point source models.
      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           point_source: ['LENSED_POSITION']

Lens Options
------------

- ``lens_option``:

  - Description: Additional options for the lens model.
  - Suboptions:

    - ``centroid_init``:

      - Description: Initial guess for the lens centroid.
      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           centroid_init: [-0.2, 0.04]

    - ``centroid_bound``:

      - Description: Half of the box width to constrain the deflector's centroid.
      - Type: ``float``
      - Default: ``0.5``
      - Example:

        .. code-block:: yaml

           centroid_bound: 0.5

    - ``limit_mass_pa_from_light``:

      - Description: Maximum allowed difference between the position angle of the mass and light profiles.
      - Type: ``float``
      - Example:

        .. code-block:: yaml

           limit_mass_pa_from_light: 10.0

    - ``limit_mass_q_from_light``:

      - Description: Maximum allowed difference between the axis ratio of the mass and light profiles.
      - Type: ``float``
      - Example:

        .. code-block:: yaml

           limit_mass_q_from_light: 0.1

Lens Light Options
------------------

- ``lens_light_option``:

  - Description: Additional options for the lens light model.
  - Suboptions:

    - ``fix``:

      - Description: Fix specific parameters for the lens light profile.
      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           fix: {0: {'n_sersic': 4.}}

    - ``gaussian_prior``:

      - Description: Gaussian priors for lens light parameters.
      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [{'param_name': 'n_sersic', 'mean': 4.0, 'sigma': 0.5}]

Source Light Options
--------------------

- ``source_light_option``:

  - Description: Additional options for the source light model.
  - Suboptions:

    - ``n_max``:

      - Description: Maximum number of Sersic profiles for each band.
      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           n_max: [4]

    - ``shapelet_scale_logarithmic_prior``:

      - Description: Whether to apply a logarithmic prior on the shapelet scale parameter.
      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           shapelet_scale_logarithmic_prior: true

Point Source Options
--------------------

- ``point_source_option``:

  - Description: Additional options for the point source model.
  - Suboptions:

    - ``ra_init``:

      - Description: Initial RA positions of the point sources.
      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           ra_init: [-0.54, -0.69, 0.19, 0.55]

    - ``dec_init``:

      - Description: Initial Dec positions of the point sources.
      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           dec_init: [-0.48, 0.54, 0.68, -0.16]

    - ``bound``:

      - Description: Bound for the point source positions.
      - Type: ``float``
      - Example:

        .. code-block:: yaml

           bound: 0.1

Fitting Options
---------------

- ``fitting``:

  - Description: Settings for the fitting process.
  - Suboptions:

    - ``pso``:

      - Description: Whether to use Particle Swarm Optimization (PSO) for fitting.
      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           pso: true

    - ``pso_settings``:

      - Description: Settings for the PSO algorithm.
      - Suboptions:

        - ``num_particle``:

          - Description: Number of particles in the swarm.
          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_particle: 20

        - ``num_iteration``:

          - Description: Number of iterations for PSO.
          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_iteration: 50

    - ``psf_iteration``:

      - Description: Whether to perform iterative PSF fitting.
      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           psf_iteration: true

    - ``psf_iteration_settings``:

      - Description: Settings for iterative PSF fitting.
      - Suboptions:

        - ``stacking_method``:

          - Description: Method for stacking PSFs.
          - Type: ``string``
          - Example:

            .. code-block:: yaml

               stacking_method: "median"

        - ``num_iter``:

          - Description: Number of PSF iterations.
          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_iter: 20

        - ``psf_iter_factor``:

          - Description: Factor for PSF iteration.
          - Type: ``float``
          - Example:

            .. code-block:: yaml

               psf_iter_factor: 0.5

        - ``keep_psf_variance_map``:

          - Description: Whether to keep the PSF variance map.
          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               keep_psf_variance_map: true

        - ``psf_symmetry``:

          - Description: Symmetry of the PSF.
          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               psf_symmetry: 4
