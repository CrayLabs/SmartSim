# Smart-Sim Library
A library for tools that aide in the convergence of simulation and AI

Modules:            (Stage)
  - Data Generation (Devel)
  - Data Processing (Devel)
  - Control         (Design)
  - Tracking        (Design)
  - ML models       (Design)
  - Visualization   (Design)
  
Data-Generation
  - Depending on the model being run, the data generation module is used to interface with the various ways to run a numerical model
    collect data from the phenomena being simulated by the model. 
  - Idea is to use ``.toml`` files to house the configurations for the models the user wants to run and then the data generation
    library handles:
        - Creating new instances of the models
        - Writing configurations into new model instances
        - Collecting simulation data

Data-Preprocessing
  - there are two primary file formats that the data-processing module will target at first: NetCDF and HDF5. Starting with NetCDF,
    each type of file format will include methods for:
       - feature selection
       - diminsionality reduction (ex. PCA)
       - data-augmentation

Control
  - In order to be able to use both AI and a numerical model in conjunction, certain aspects of the workload manager and allocation strategy need to be created and monitored carefully as the two don't always play nicely with each other.
  - The control module interfaces with the workload manager, numerical "mpi-run" models, and ML models. 
  - The vision for the control module is to be able to do everything from simple starting and stopping to in-situ training and inference.

Tracking
  - This module sits below all the other modules within the smart-sim library. It tracks and logs everything in the workflow of using a numerical model. 
  - Tracked items include:
       - Configurations
       - Simulation output
       - Hyperparameters for ml-models
       - HPO results

ML Models
  - This module contains all of the models used for Smart-Sim.
  - A template interface will be defined so that users can drop their models into the module and use them with ease.


Visualization
  - In situ visualization is necessary for developers to understand what is going on inside of a model.
  - The ablity to visualize features in real-time presents a unqiue opportunity as the next best, commonly used tool is atiquated(ncview)

