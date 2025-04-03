Quickstart Guide
================

This guide provides step-by-step instructions on how to set up the ``dolphin`` ecosystem for full automation with AI and generate model overview plots.

Setting up the ``dolphin`` Ecosystem
----------------------------------

1. **Install** ``dolphin`` **and dependencies**:
    Ensure you have installed ``dolphin``, ``lenstronomy``, and all required dependencies. You can install ``dolphin`` using pip:

    .. code-block:: bash

        pip install space-dolphin

2. **Create an input/output directory**:
    Create a directory to serve as the input/output directory for ``dolphin``. For an example, check the ``io_directory_example``. The directory can have any name.

3. **Set up the directory structure**:
    Inside the input/output directory, create the following subdirectories; these subdirectory names are fixed and should not be changed:

    - **data**: Contains subdirectories for each lens system. Each subdirectory should include:
      - Image data files (in HDF5 format).
      - PSF files (in HDF5 format).
    - **settings**: Contains configuration files (``config_{lens_name}.yaml``) for each lens system.
    - **masks**: *(Optional)* Contains custom mask files (``mask_{lens_name}_{band}.npy``) for each lens system.
    - **logs**: Stores log files generated during model runs.
    - **outputs**: Saves the model outputs.
    - **hpc**: *(Optional)* Contains scripts for submitting batch jobs in MPI environments.

    Example directory structure:

    .. code-block:: text

        io_directory_example/
        ├── data/
        │   ├── system_name/
        │   │   ├── image_system_name_band.h5 (provided by user)
        │   │   ├── psf_system_name_band.h5 (provided by user)
        ├── settings/
        │   ├── config_system_name.yaml (can be created by AI)
        │   ├── masks/
        │   │   ├── mask_system_name_band.npy (can be created by AI)
        ├── logs/
        ├── outputs/
        ├── hpc/

Content of image data and PSF files
-----------------------------------

The image data and the PSF file must be in the HDF5 format. The keyword/datasets contained in these files are in the convention (keyword naming) of ``lenstronomy``.

The image data file needs to have the following datasets:

- ``image_data``: reduced and background-subtracted image cutout centered at the lens system,
- ``background_rms``: background level,
- ``exposure_time``: the map of exposure times for each pixel, so that ``image_data * exposure_time`` is Poisson noise distributed,
- ``ra_at_xy_0``: RA of the (0, 0) pixel in the ``image_data`` cutout,
- ``dec_at_xy_0``: Dec of the (0, 0) pixel in the ``image_data`` cutout,
- ``transform_pix2angle``: a transform matrix to map the pixel numbers (x, y) to angles (RA, Dec).

The PSF data file needs to have the following datasets:

- ``kernel_point_source``: a pixelated PSF (not required to have the same dimension of ``image_data``),
- ``psf_variance_map``: *optional*, uncertainty in the provided PSF, needs to have the same dimension of ``kernel_point_source``.

Running ``dolphin`` with Full Automation
-------------------------------------

Use the following Python code to run the ``dolphin`` pipeline for a specific lens system. For example, to model a lensed quasar system:

.. code-block:: python

    from dolphin.ai import Vision
    from dolphin.ai import Modeler
    from dolphin.processor import Processor

    io_directory_path = "path/to/io_directory"

    # Step 1: Create segmentation for the lens system
    vision = Vision(io_directory_path, source_type="quasar")
    vision.create_segmentation_for_single_lens(
         lens_name="system_name", band_name="filter_name"
    )

    # Step 2: Create configuration for the lens system
    modeler = Modeler(io_directory_path)
    modeler.create_config_for_single_lens(
         lens_name="system_name", band_name="filter_name"
    )

    # Step 3: Run the model
    processor = Processor(io_directory_path)
    processor.swim(
         lens_name="system_name", model_id="example", recipe_name="galaxy-quasar"
    )

Replace ``system_name`` and ``filter_name`` with the appropriate lens name and band name for your system. The ``source_type`` parameter in the ``Vision`` class can be set to either ``quasar`` or ``galaxy`` depending on the type of lens system being modeled.

Check the outputs: After running the pipeline, check the ``logs/`` directory for log files and the ``outputs/`` directory for the model outputs.

Generate an overview plot: To visualize the results, use the following Python code to generate a model overview plot:

.. code-block:: python

    from dolphin.analysis import Output

    output = Output(io_directory_path)
    fig = output.plot_model_overview(
         lens_name="system_name", model_id="example"
    )

This will create a plot summarizing the lens model. You can save the plot to a file using ``fig.savefig()``.

Congratulations! You have successfully set up and run ``dolphin`` for full automation with AI and generated a model overview plot. 

For more examples in Jupyter notebooks, check out the ``notebooks`` `folder <https://github.com/ajshajib/dolphin/tree/main/notebooks>`_. Additionally, check out `"Project Dinos" GitHub repo <https://github.com/Project-Dinos/dinos-i/tree/main/2_dolphin_modelling>`_ for lots of examples of manually produced ``config.yaml`` files that were used in the semi-automated modeling scheme for the Dinos sample in `Tan et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.1474T/abstract>`_. As a flexible modeling workflow, AI-generated configs can be tweaked by hand before fitting the model(s) for additional improvements and modifications in the model setup.

