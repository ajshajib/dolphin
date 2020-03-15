#!/usr/bin/env bash

# (USER) SETTINGS
nodes=1
slots=24
time=24 # this will give you 24 hour runtime
log_dir="/u/home/username/logs" # set a log directory
email="" # provide your email address for job notifications
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

cat << EOF > ./${run_id}_${lens_name}.sh
#!/bin/bash
#SBATCH -A cla199
#SBATCH --job-name="${run_id}_${lens_name}"
#SBATCH --output=${log_dir}"/joblog.%j"
#SBATCH --partition=compute
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${slots}
#SBATCH --export=ALL
#SBATCH --qos=oneweek
#SBATCH -t ${time}:00:00
#SBATCH --mail-user=${email}
#SBATCH --mail-type=ALL

echo ""
echo "Job (run_sequence.py) $SLURM_JOB_ID started on:   "` hostname -s `
echo "Job (run_sequence.py) $SLURM_JOB_ID started on:   "` date `
echo ""

# (USER) ENVIRONMENTS, load and set environments
module load intel/2016.3.210
module load intelmpi/2016.3.210
export OMP_NUM_THREADS=1
#module load python/3.7.2
# END ENVIRONMENTS

echo ""
module list
echo ""

which mpirun
which python
echo""

#
# Run the user program
#

echo "`which mpirun` -np $((nodes*slots))  `which python` $base_dir/swim.py
        $lens_name $run_id >& $log_dir/output.\$SLURM_JOB_ID"
time `which mpirun` -np $((nodes*slots)) `which python` $base_dir/swim.py \
        $lens_name $run_id >& $log_dir/output.\$SLURM_JOB_ID


echo ""
echo "job (run_sequence.py) \$SLURM_JOB_ID  finished at:  " ` date `
echo ""

EOF

chmod u+x ${run_id}_${lens_name}.sh

if [[ -x ${run_id}_${lens_name}.sh ]]; then
    echo "sbatch ${run_id}_${lens_name}.sh"
    sbatch ${run_id}_${lens_name}.sh
fi