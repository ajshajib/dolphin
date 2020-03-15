#!/usr/bin/env bash

# (USER) SETTINGS
slots=16 # number of cores
mem=2000 # memory, Megabyte per core
time=24 # number of hours
hp="" # flag for private/public nodes, for public: "", for private: ",highp"
log_dir="/u/home/username/logs" # set a log directory
# END SETTINGS

function usage {
    echo -e "\nUsage:\n $0 <config_string>"
}

if [ $# == 0 ]; then
    echo -e "\n Please provide a config_string"
    usage
    exit
fi

run_id="$1"
lens_name="$2"

base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cat << EOF > ./${run_id}_${lens_name}.cmd
#!/bin/bash
#  UGE job for run_sequence.py built Thu Feb 16 09:35:24 PST 2017
#
#  The following items pertain to this script
#  Use current working directory
#$ -cwd
#  input           = /dev/null
#  output          = $log_dir/joblog
#$ -o $log_dir/joblog.\$JOB_ID
#  error           = Merged with joblog
#$ -j y
#  The following items pertain to the user program
#  user program    = $base_dir/swim.py
#  arguments       = $run_id, $lens_name
#  program input   = Specified by user program
#  program output  = Specified by user program
#  Parallelism:  $slots-way parallel
#  Resources requested
#$ -pe dc_* $slots
#$ -l h_data=${mem}M,h_rt=${time}:00:00$hp
#
#$ -M $USER@mail
#  Notify at beginning and end of job
#$ -m bea
#  Job is not rerunable
#$ -r n
#  Uncomment the next line to have your environment variables used by SGE
# -V
#
#
echo ""
echo "Job (swim.py) \$JOB_ID started on:   "\` hostname -s \`
echo "Job (swim.py) \$JOB_ID started on:   "\` date \`
echo ""

# (USER) ENVIRONMENTS, load and set environments
. /u/local/Modules/default/init/modules.sh
export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=4
module load python/3.7.2
#export PYTHONPATH=$HOME/python_packages/lib/python3.7/site-packages:$HOME/python_packages:$PYTHONPATH
# END ENVIRONMENTS

module list
which mpirun
which python3

#
# Run the user program
#

echo "\`which mpirun\` -np ${slots} \`which python3\` $base_dir/swim.py
          $lens_name $run_id >& $log_dir/output.\$JOB_ID"

time \`which mpirun\` -np ${slots} \`which python3\`  \\
          $base_dir/swim.py $lens_name $run_id >& $log_dir/output.\$JOB_ID


echo ""
echo "job (swim.py) \$JOB_ID  finished at:  "` date `
echo ""

EOF

chmod u+x ${run_id}_${lens_name}.cmd

if [[ -x ${run_id}_${lens_name}.cmd ]]; then
    echo "qsub ${run_id}_${lens_name}.cmd"
    qsub ${run_id}_${lens_name}.cmd
fi