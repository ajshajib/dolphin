Configuration Files
===================

This document provides a detailed explanation of all the allowed options in the ``config.yaml`` files for ``dolphin``, but some of them are optional, as indicated. Check out the ``io_directory_example/settings`` `folder <https://github.com/ajshajib/dolphin/tree/main/io_directory_example/settings>`_ for some example config files.

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

``dolphin`` supports two ways of producing per-band likelihood masks: programmatically from parameters specified in the settings, or by loading user-provided mask files. When both are present, the user-provided mask files take precedence.

- ``mask``: *(Optional)* Settings for masking regions of the image.

  - Suboptions:
    
    - ``provided``: Whether to load the mask from a ``.npy`` file instead of generating it. When ``True``, ``dolphin`` loads ``{io_directory}/settings/masks/mask_{lens_name}_{band}.npy`` for each band listed in ``band`` and ignores all other mask suboptions. See :ref:`Providing a mask file <providing-a-mask-file>` for details. When ``False`` or absent, the mask is generated from the other suboptions below.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           provided: True

    - ``centroid_offset``: Offset for the centroid of the mask, in arcsec, relative to the deflector center. One ``[ra_offset, dec_offset]`` pair per band.

      - Type: ``list of lists of floats``
      - Example:

        .. code-block:: yaml

           centroid_offset: [[0.0, 0], [0.0, 0]]

    - ``mask_edge_pixels``: Number of edge pixels to mask on each side of the image. One value per band.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           mask_edge_pixels: [0, 2]

    - ``radius``: Radius of a circular mask for each band, in arcsec. Pixels inside the circle are kept; pixels outside are masked. Mutually exclusive with ``a``/``b``/``angle``.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           radius: [20.0, 20.0]

    - ``a``, ``b``, ``angle``: Elliptical mask parameters (semi-major axis, semi-minor axis, and rotation angle in radians). All three must be provided together, one value per band. Use this instead of ``radius`` when an elliptical mask is needed.

      - Type: ``list of floats`` (each)
      - Example:

        .. code-block:: yaml

           a: [2.5]
           b: [1.8]
           angle: [0.0]

    - ``extra_regions``: Additional circular regions to mask out (e.g., nearby companions or satellites). For each band, provide a list of ``[ra_offset, dec_offset, radius]`` triplets, where offsets are in arcsec relative to the deflector center and ``radius`` is in arcsec.

      - Type: ``list of lists of lists of floats``
      - Example:

        .. code-block:: yaml

           extra_regions: [[[1.2, -0.5, 0.3], [-0.8, 0.9, 0.2]]]

Providing a mask file
~~~~~~~~~~~~~~~~~~~~~

When ``mask.provided`` is ``True``, ``dolphin`` loads one ``.npy`` file per band from the ``masks/`` subdirectory of the settings directory:

- **File location**: ``{io_directory}/settings/masks/``.
- **Filename pattern**: ``mask_{lens_name}_{band}.npy``.
- **Array shape**: must match the ``image_data`` cutout shape.
- **Convention**: ``1`` = pixel is included in the likelihood, ``0`` = pixel is masked out.

An example mask is included at ``io_directory_example/settings/masks/mask_lensed_quasar_F814W.npy``.

The example below loads `lens_system3`, searches outside the central region where the strong lens is for bright clumps, and places a circular mask over each such contaminant:

.. code-block:: python

  import h5py
  import numpy as np
  from scipy.ndimage import binary_dilation, label

  # Load an image and background RMS.
  with h5py.File(
      "../io_directory_example/data/lens_system3/image_lens_system3_F475X.h5"
  ) as f:
      image = f["image_data"][:]
      background_rms = float(f["background_rms"][()])

  num_pixel = image.shape[0]  # e.g., 130 in the case of lens_system3_F475X
  y, x = np.indices(image.shape)
  r = np.hypot(x - num_pixel / 2, y - num_pixel / 2)

  # Find bright pixels (> 2 sigma) outside the central 40-pixel-radius region.
  bright_outside = (image > 2 * background_rms) & (r > 40)

  # Keep only connected components of 16 or more pixels
  labels, _ = label(bright_outside)
  sizes = np.bincount(labels.ravel())
  sizes[0] = 0  # ignore the background label
  contaminants = np.isin(labels, np.flatnonzero(sizes >= 16))

  # Expand each contaminant into a 4-pixel-radius circular exclusion zone.
  disk = np.hypot(*(np.indices((9, 9)) - 4)) <= 4
  exclusion = binary_dilation(contaminants, structure=disk)

  # 1 = kept, 0 = masked.
  mask = np.where(exclusion, 0.0, 1.0)

  np.save("../io_directory_example/settings/masks/mask_lens_system3_F475X.npy", mask)
