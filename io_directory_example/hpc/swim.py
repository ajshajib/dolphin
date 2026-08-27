import os
import sys
import time

import schwimmbad

from dolphin.processor import Processor

pool = schwimmbad.choose_pool(mpi=True)

start_time = time.perf_counter()

cwd = os.getcwd()
base_path, _ = os.path.split(cwd)

processor = Processor(base_path)

lens_name = str(sys.argv[1])
model_id = str(sys.argv[2])

if pool.is_master():
    print(f"Run [{model_id}] for {lens_name} loaded.")

processor.swim(lens_name, model_id=model_id)

if pool.is_master():
    end_time = time.perf_counter()
    print(f"Total time needed for computation: {end_time - start_time:.2f} s")
