

<div align="center">
    <a href="https://github.com/CrayLabs/SmartSim"><img src="https://raw.githubusercontent.com/CrayLabs/SmartSim/master/doc/images/SmartSim_Large.png" width="90%"><img></a>
    <br />
    <br />
<div display="inline-block">
    <a href="https://github.com/CrayLabs/SmartSim"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.craylabs.org/docs/installation.html"><b>Install</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.craylabs.org/docs/overview.html"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://join.slack.com/t/craylabs/shared_invite/zt-nw3ag5z5-5PS4tIXBfufu1bIvvr71UA"><b>Slack Invite</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/CrayLabs"><b>Cray Labs</b></a>&nbsp;&nbsp;&nbsp;
  </div>
    <br />
    <br />
</div>


[![License](https://img.shields.io/github/license/CrayLabs/SmartSim)](https://github.com/CrayLabs/SmartSim/blob/master/LICENSE.md)
![GitHub last commit](https://img.shields.io/github/last-commit/CrayLabs/SmartSim)
![GitHub deployments](https://img.shields.io/github/deployments/CrayLabs/SmartSim/github-pages?label=doc%20build)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/CrayLabs/SmartSim)
![Language](https://img.shields.io/github/languages/top/CrayLabs/SmartSim)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


------------

# SmartSim

SmartSim makes it easier to use common Machine Learning (ML) libraries
like PyTorch and TensorFlow, in High Performance Computing (HPC) simulations
and workloads.

SmartSim provides an API to connect HPC (MPI + X) simulations
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
| ONNX       |       1.2.0       |

At this time, PyTorch is the most tested within SmartSim and we recommend
users use PyTorch at this time if possible.


SmartSim is made up of two parts
  1. SmartSim Infrastructure Library (This repository)
  2. [SmartRedis](https://github.com/CrayLabs/SmartRedis)

## SmartSim Infrastructure Library

The Infrastructure Library (IL) helps users get the Orchestrator running
on HPC systems. In addition, the IL provides mechanisms for creating, configuring,
executing and monitoring HPC workloads. Users can launch everything needed
to run converged ML and simulation workloads right from a jupyter
notebook using the IL Python interface.

### Dependencies

The following third-party (non-Python) libraries are used in the SmartSim IL.

 - [Redis](https://github.com/redis/redis) 6.0.8
 - [RedisAI](https://github.com/RedisAI/RedisAI) 1.2
 - [RedisIP](https://github.com/Spartee/RedisIP) 0.1.0

## SmartRedis

The SmartSim IL Clients ([SmartRedis](https://github.com/CrayLabs/SmartRedis))
are implementations of Redis clients that implement the RedisAI
API with additions specific to scientific workflows.

SmartRedis clients are available in Fortran, C, C++, and Python.
Users can seamlessly pull and push data from the Orchestrator from different languages.

| Language 	| Version/Standard 	|
|----------	|:----------------:	|
| Python   	|        3.7+      	|
| Fortran  	|       2003       	|
| C        	|        C99       	|
| C++      	|       C++11      	|

SmartRedis clients are cluster compatible and work with the OSS Redis/RedisAI stack.

### Dependencies

SmartRedis utilizes the following libraries.

 - [NumPy](https://github.com/numpy/numpy)
 - [Hiredis](https://github.com/redis/hiredis) 1.0.0
 - [Redis-plus-plus](https://github.com/sewenew/redis-plus-plus)  1.2.3
 - [protobuf](https://github.com/protocolbuffers/protobuf)  v3.11.3


## Publications

The following are public presentations or publications using SmartSim (more to come!)

 - [Collaboration with NCAR - CGD Seminar](https://www.youtube.com/watch?v=2e-5j427AS0)
 - [Using Machine Learning in HPC Simulations - paper (pre-print)](https://arxiv.org/abs/2104.09355)


## Cite

Please use the following citation when referencing SmartSim, SmartRedis, or any SmartSim related work.


Partee et al., “Using Machine Learning at Scale in HPC Simulations with SmartSim:
An Application to Ocean Climate Modeling,” arXiv:2104.09355, Apr. 2021,
[Online]. Available: http://arxiv.org/abs/2104.09355.

### bibtex

    ```latex
    @misc{partee2021using,
          title={Using Machine Learning at Scale in HPC Simulations with SmartSim: An Application to Ocean Climate Modeling},
          author={Sam Partee and Matthew Ellis and Alessandro Rigazzi and Scott Bachman and Gustavo Marques and Andrew Shao and Benjamin Robbins},
          year={2021},
          eprint={2104.09355},
          archivePrefix={arXiv},
          primaryClass={cs.CE}
    }
    ```
