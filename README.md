# Smart-Sim Library

    A library of tools dedicated to accelerating the convergence of AI and numerical
    simulation models. SmartSim can connect models written in Fortran, C, C++ and
    Python to the modern data science stack. Integration with workload managers like
    Slurm make it easy to run mutliple jobs for simulation, analysis, and visualization
    all within a single allocation. Generate configurations and run ensembles of
    simulations all within the comfort of a jupyter notebook.

## Current Features

   - Clients in Python, C, C++ and Fortran
   - Model Parameter Optimization (MPO) with CrayAI
   - Allocation management interface through Slurm
   - Ensembling through text-based configuration generation for models
   - Works on compute nodes for rapid prototyping and preprocessing
   - Runs inside Jupyter lab/notebook
   - Compatiable with Urika Data Analytics Stack (Spark, Dask, etc)

## Setup

   - Clone the git repository
      > git clone {insert git address} Smart-Sim
   - Set SmartSim env variables and add to python path
      > cd Smart-Sim && source setup_env.sh
   - Install Dependencies
      > pip install -r requirements.txt
