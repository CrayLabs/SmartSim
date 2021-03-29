<div align="center">
  <img src="https://github.com/CrayLabs/SmartSim/blob/cleanup/doc/images/ss-green-logo.png" width="40%"><img>
</div>


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

------------

# SmartSim

SmartSim makes it easier to use common Machine Learning (ML) libraries
like PyTorch and TensorFlow, on numerical simulations at scale.

Essentially, SmartSim provides an API to connect HPC (MPI + X) simulations
written in Fortran, C, C++, and Python to an in-memory database called
the Orchestrator. The Orchestrator is built on Redis, a popular caching
database written in C. This connection between simulation and database
is the fundamental paradigm of SmartSim. Simulations in the aforementioned
languages can stream data to the Orchestrator and pull the data out
in Python for online analysis, visualization, and training.

In addition, the Orchestrator is equipped with ML inference runtimes:
PyTorch, TensorFlow, and ONNX. From inside a simulation, users can
store and execute trained models and retrieve the result.


### Supported ML Libraries

SmartSim 0.3.0 uses Redis 6.0.8 and RedisAI 1.2

| Library    | Supported Version |
|------------|:-----------------:|
| PyTorch    |       1.7.0       |
| TensorFlow |       1.15.0      |
| TFLite     |       2.0.0       |
| ONNX       |       1.2.0       |

At this time, PyTorch is the most tested within SmartSim and we recommend
users use PyTorch at this time if possible.



SmartSim is made up of two parts
  1. SmartSim Infrastructure Library
  2. SmartSim Infrastructure Library Clients (SILC)

## SmartSim Infrastructure Library

The Infrastructure Library (IL) helps users get the Orchestrator running
on HPC systems. In addition, the IL provides mechanisms for creating, configuring,
executing and monitoring simulations. Users can launch everything needed
to run converged ML and simulation worklaods right from a jupyter
notebook using the IL Python interface.

## SILC

The SmartSim Infrastructure Library Clients are implementations of Redis
clients that implement the RedisAI API with a few additions specific to
HPC simulations.

SILC clients are available in Fortran, C, C++, and Python. Users can seemlessly
pull and push data from the Orchestrator from different langauges.

| Language 	| Version/Standard 	|
|----------	|:----------------:	|
| Python   	|    3.7 - 3.8.5   	|
| Fortran  	|       2003       	|
| C        	|        C99       	|
| C++      	|       C++11      	|

The SILC clients are cluster compatible and work with the OSS Redis/RedisAI stack.

