Quickstart Guide
================

This guide provides step-by-step instructions to set up the `dolphin` ecosystem for full automation with AI and generate model overview plots.

Setting up the `dolphin` Ecosystem
----------------------------------

1. **Install `dolphin` and dependencies**:  
    Ensure you have installed `dolphin`, `lenstronomy`, and all required dependencies. You can install `dolphin` using pip:

```bash
    pip install space-dolphin
```

2. **Create an input/output directory**:  
    Create a directory to serve as the input/output directory for `dolphin`. For an example, check the `io_directory_example`. The directory can have any name.

3. **Set up the directory structure**:  
    Inside the input/output directory, create the following subdirectories. These subdirectory names are fixed and should not be changed:

    - **data**: Contains subdirectories for each lens system. Each subdirectory should include:
      - Image data files (in HDF5 format).
      - PSF files (in HDF5 format).
    - **settings**: Contains configuration files (`config_{lens_name}.yml`) for each lens system.
    - **masks**: *(Optional)* Contains custom mask files (`mask_{lens_name}_{band}.npy`) for each lens system.
    - **logs**: Stores log files generated during model runs.
    - **outputs**: Saves the model outputs.
    - **hpc**: *(Optional)* Contains scripts for submitting batch jobs in MPI environments.

    Example directory structure:

    ```
    io_directory_example/
    ├── data/
    │   ├── ai_test/
    │   │   ├── image_data.h5
    │   │   ├── psf_data.h5
    ├── settings/
    │   ├── config_ai_test.yml
    ├── masks/
    │   ├── mask_ai_test_band1.npy
    ├── logs/
    ├── outputs/
    ├── hpc/
    ```

Content of Image Data and PSF Files
-----------------------------------

The image data and the PSF file need to be in the HDF5 format. The contained keywords/datasets in these files follow the conventions (keyword naming) of `lenstronomy`.

The image data file needs to have the following datasets:

- `image_data`: Reduced and background-subtracted image cutout centered at the lens system.
- `background_rms`: Background level.
- `exposure_time`: The map of exposure times for each pixel, so that `image_data * exposure_time` is Poisson noise distributed.
- `ra_at_xy_0`: RA of the (0, 0) pixel in the `image_data` cutout.
- `dec_at_xy_0`: Dec of the (0, 0) pixel in the `image_data` cutout.
- `transform_pix2angle`: A transform matrix to map the pixel numbers (x, y) to angles (RA, Dec).

The PSF data file needs to have the following datasets:

- `kernel_point_source`: A pixelated PSF (not required to have the same dimension as `image_data`).
- `psf_variance_map`: *(Optional)* Uncertainty in the provided PSF, needs to have the same dimension as `kernel_point_source`.

Running `dolphin` for Full Automation
-------------------------------------

Use the following Python code to run the `dolphin` pipeline for a specific lens system. For example, to model the quadruply lensed quasar system J2205-3727:

