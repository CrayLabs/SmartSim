# Smart-Sim Library
A library for tools that aide in the convergence of simulation and AI

Modules:
  - Data Generation
  - Data Processing
  - ML models
  - Inference Interface
  
Data-Generation
  - Depending on the model being run, the data generation module is used to interface with the various ways to run a numerical model
    collect data from the phenomena being simulated by the model. 
  - Idea is to use ``.toml`` files to house the configurations for the models the user wants to run and then the data generation
    library handles:
        - Creating new instances of the models
        - Writing configurations into new model instances
        - Interfacing with workload managers or underlying model run framework(ex. cime)
        - Collecting simulation data

Data-Preprocessing
  - there are two primary file formats that the data-processing module will target at first: NetCDF and HDF5. Starting with NetCDF,
    each type of file format will include methods for:
       - feature selection
       - diminsionality reduction (ex. PCA)
       - data-augmentation

ML Models
  - This module contains all of the models used for Smart-Sim.
  - A template interface will be defined so that users can drop their models into the module and use them with ease.

Inference Interface
  - The inference module contains the objects that result from the use of the library. This module will be what users can directly
    query for a number of different use cases.
    
    
High Level Feature Goals
  - Find optimal parameters for a model (MPO)
  - Auto-enhance simulation output
  - Real-time data ingestion and inference (for things like anamoly detection)
  - Full model emulation (constained GAN)
  - in situ simulation guidance (MPO on the fly)
