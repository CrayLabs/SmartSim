#!/bin/bash -x
#PBS -N resv_job
#PBS -l select=1:ncpus=36:mpiprocs=36+1:ncpus=1:mpiprocs=1:ngpus=2
#PBS -l gpu_type=v100
#PBS -l walltime=00:30:00
#PBS -W create_resv_from_job=true
#PBS -j oe
#PBS -k oed
#PBS -q casper
#PBS -A P93300606

for rsv in $(qstat -Q|awk '$1 ~ /^R/{print $1}')
do
   parent_job=$(pbs_rstat -F $rsv|awk '$1 ~ /^reserve_job/{print $3}')
   if [[ "${PBS_JOBID}" == "${parent_job}" ]] ; then
      rsvname=$rsv
      break
   fi
done
if [ -z $rsvname ]; then echo "rsv is unset"; exit -1; else echo "rsv name is set to '$rsvname'"; fi

me=$(whoami)
pbs_ralter -U $me $rsvname
export CESM_ROOT=/glade/work/jedwards/sandboxes/cesm2_x_alpha.smartsim/
gpu_jobid=$(qsub -q $rsvname -v CESM_ROOT launch_database_cluster.py)

head_host=$(qstat -f $PBS_JOBID|awk '$1 ~ /^exec_host$/{print $3}'|cut -d\/ -f1-1)
SSDB="$(getent hosts ${head_host}-ib|awk '{print $1}'):6780"
export SSDB
#./xmlchange JOB_QUEUE=$rsvname --subgroup case.test --force
#./xmlchange JOB_WALLCLOCK_TIME=00:20:00 --subgroup case.test
#./case.submit
qsub -l walltime=00:20:00 -AP93300606 -q $rsvname -v SSDB ./smartredis_put_get_3D
# clean up

#
#pbs_rdel $rsvname
cat <<EOF1 > cleanup.cmd
#!/bin/bash
#PBS -N cleanup
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -k oed
#PBS -A P93300606
do
  running=\$(qstat -Q $rsvname | awk '\$6 ~/[0-9]+/print {\$6, \$7}'}
  if [[ "\$running" == "0 0" ]] ; then
    pbs_rdel $rsvname
    break
  fi
  sleep 10
done
EOF1
qsub -q casper ./cleanup.cmd
