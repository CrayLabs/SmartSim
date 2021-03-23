
Orchestrator
============

.. currentmodule:: smartsim.database

Local Orchestrator
------------------

The local Orchestrator can be launched through
the local launcher and does not support cluster
instances

.. autoclass:: Orchestrator
   :members:
   :inherited-members:
   :undoc-members:

PBSPro Orchestrator
-------------------

The PBSPro Orchestrator can be launched as a batch, and
in an interactive allocation.

.. autoclass:: PBSOrchestrator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Slurm Orchestrator
------------------

The Slurm Orchestrator is used to launch Redis on to Slurm WLM
systems and can be launched as a batch, on existing allocations,
or in an interactive allocation.

.. autoclass:: SlurmOrchestrator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

