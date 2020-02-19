# -*- coding: utf-8 -*-

import os
import sys
from dolphin.processor import Processor
import time
from mpi4py import MPI


comm = MPI.COMM_WORLD

start_time = time.time()

cwd = os.getcwd()
base_path, _ = os.path.split(cwd)

processor = Processor(base_path)

lens_name = str(sys.argv[1])
model_id = str(sys.argv[2])

if comm.Get_rank() == 0:
    print("Run [{}] for {} loaded.".format(model_id, lens_name))

processor.swim(lens_name, model_id=model_id)

if comm.Get_rank() == 0:
    end_time = time.time()
    print('Total time needed for computation: {:.2f} s'.format(end_time -
                                                               start_time))