# Casper

```bash
module purge
module use /glade/p/cesmdata/cseg/PROGS/modulefiles/CrayLabs
module load gnu ncarcompilers openmpi netcdf ncarenv cmake
module load SmartRedis
```

I also needed a newer version of gmake, it's in /glade/work/jedwards/make-4.3/bin/make

I am using a python environment created with:
```
ncar_pylib -c 20201220 /glade/work/$USER/casper_npl_clone
```

``pip install smartsim``

``smart --device gpu``

``pip install smartredis``

First you need to build the smartredis_put_get_3D.F90 fortran example:
```
make
```

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
It creates pbs jobs from each of the 3 templates
1. resv_job.template
2. launch_database_cluster.template
3. launch_client.template

and submits the resv_job.sh which in turn will create a reservation large enough for the db and all the ensemble members.
It submits those jobs in the newly created reservation.  It starts the database and sets the SSDB environment variable
then launchs each of the clients, all of this is done within the newly created reservation.   The database job monitors progress of the clients and exits and removes the reservation when it is complete.

Note that this launches the database and client jobs separately - The prefered method is to launch the client through SmartSim. 

** Currently to use this feature you must first send a note to cislhelp@ucar.edu and ask for permission to use the
create_resv_from_job feature of PBS.  **