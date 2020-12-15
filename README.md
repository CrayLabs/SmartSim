# Smart-Sim Library

    A library of tools dedicated to accelerating the convergence of AI and numerical
    simulation models. SmartSim can connect models written in Fortran, C, C++ and
    Python to the modern data science stack. Integration with workload managers like
    Slurm make it easy to run multiple jobs for simulation, analysis, and visualization
    all within a single allocation. Generate configurations and run ensembles of
    simulations all within the comfort of a jupyter notebook.

## Current Features

   - Clients in Python, C, C++ and Fortran (SILC)
   - Allocation management interface through Slurm
   - Ensembling through text-based configuration generation for models
   - Works on compute nodes for rapid prototyping and preprocessing
   - Runs inside Jupyter lab/notebook
   - Distributed, in-memory database
   - Pytorch, Tensorflow, and ONNX based inference suppport with RedisAI


## Documentation

   - To build the documentation, clone the repo, install the ``requirements-dev.txt``
     and execute ``make docs``. Then open ``doc/_build/html/index.html`` in a browser.

