# -*- coding: utf-8 -*-
"""
This script submits an array of jobs to a HPC cluster. Run using the command ::

    $ python submit_jobs.py run_id

This script retrieves the lens list from ../lens_list.txt file and then sends
to create_job_sge.sh or create_job_slurm.sh to submit a job for each lens.

**Remember** to set the correct value in the `job_system` variable below. You
also need to modify the corresponding `create_job_{}.sh` file to set the correct
computational settings (e.g., number of cores, requested hours), and to load
the appropriate environment for running the jobs. These settings are marked by

    # (USER)
    [...]
    # End

tags in the `create_job_{}.sh` files.
"""
__author__ = "ajshajib"

import sys
import os
import time

from dolphin.processor.files import FileSystem

job_system = "sge"  # 'sge' or 'slurm'

try:
    str(sys.argv[1])
except IndexError:
    print("run_id needed: python submit_jobs.py run_id")
else:
    run_id = str(sys.argv[1])  # identifier for modeling run

    cwd = os.getcwd()
    base_path, _ = os.path.split(cwd)

    file_system = FileSystem(base_path)

    lens_list = file_system.get_lens_list()

    for lens_name in lens_list:
        os.system("./create_job_{}.sh {} {}".format(job_system, run_id, lens_name))
        print("./create_job_{}.sh {} {}".format(job_system, run_id, lens_name))
        time.sleep(1)

    print("{} jobs submitted!".format(len(lens_list)))
