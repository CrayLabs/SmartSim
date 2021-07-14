
# Casper 

```bash
module purge
module load gnu/9.1.0 ncarcompilers openmpi netcdf ncarenv cmake
```

I also needed a newer version of gmake, it's in /glade/work/jedwards/make-4.3/bin/make

I am using a python environment created with:
```
ncar_pylib -c 20201220 /glade/work/$USER/casper_npl_clone
```

``pip install smartsim``
``smart --device gpu``
``pip install smartredis``

launch.py is the primary launch script 
```
usage: launch.py [-h] [--db-nodes DB_NODES] [--ngpus-per-node NGPUS_PER_NODE]
                 [--walltime WALLTIME] [--ensemble-size ENSEMBLE_SIZE]
                 [--member-nodes MEMBER_NODES] [--account ACCOUNT]
                 [--db-port DB_PORT]

optional arguments:
  -h, --help            show this help message and exit
  --db-nodes DB_NODES   Number of nodes for the SmartSim database, default=1
  --ngpus-per-node NGPUS_PER_NODE
                        Number of gpus per SmartSim database node, default=0
  --walltime WALLTIME   Total walltime for submitted job, default=00:30:00
  --ensemble-size ENSEMBLE_SIZE
                        Number of ensemble members to run, default=1
  --member-nodes MEMBER_NODES
                        Number of nodes per ensemble member, default=1
  --account ACCOUNT     Account ID
  --db-port DB_PORT     db port, default=6780
```
