# -*- coding: utf-8 -*-

import os
import sys
from dolphin.processor import Processor
import time
import schwimmbad


pool = schwimmbad.choose_pool(mpi=True)

start_time = time.perf_counter()

cwd = os.getcwd()
base_path, _ = os.path.split(cwd)

processor = Processor(base_path)

lens_name = str(sys.argv[1])
model_id = str(sys.argv[2])

if pool.is_master():
    print("Run [{}] for {} loaded.".format(model_id, lens_name))

processor.swim(lens_name, model_id=model_id)

if pool.is_master():
    end_time = time.perf_counter()
    print('Total time needed for computation: {:.2f} s'.format(end_time -
                                                               start_time))