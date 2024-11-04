**************
LaunchSettings
**************
========
Overview
========
The ``LaunchSettings`` class manages launcher configuration settings to support launcher-specific behavior in jobs.
SmartSim supports multiple launchers and allows users to set launch arguments and environment variables.
Additionally, the ``LaunchSettings`` class provides methods to access and configure launch arguments and
environment variables.

**Dragon** is a launcher which that enables fast execution of complex workflows on HPC systems. It works within
a Slurm allocation, making it easy to run large workflows efficiently on HPC machines.

===================
Supported Launchers
===================
The ``LaunchSettings`` class supports multiple launchers, each customized for different environments. Among the launchers,
**Dragon** stands out as fast and ideal for large-scale HPC jobs. The following launchers are categorized
based on the systems or programs they are designed for.

HPE Cray
========
SmartSim provides support for two HPE Cray system launchers: **ALPS** and **PALS**. ALPS is designed for legacy Cray platforms
to address both future Cray platform needs and the limitations of older software components. PALS acts as a launcher for third-party
workload managers (WLMs) that do not have their own launchers.

.. list-table:: HPE Cray Specific Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``aprun``
     - Application Level Placement Scheduler (ALPS).
   * - ``mpiexec``
     - Parallel Application Launch Service (PALS).


IBM
===
SmartSim provides support for an IBM launcher: **LSF**. IBM's LSF is a platform for managing workloads and scheduling jobs
in distributed high-performance computing (HPC) environments.

.. list-table:: IMB Specific Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``jsrun``
     - Load Sharing Facility (LSF).

Standard MPI
============
SmartSim supports three commands to launch Message Passing Interface (MPI) applications: ``mpiexec``, ``mpirun`` and ``orterun``.
Each execute serial and parallel jobs in Open MPI.

.. list-table:: MPI Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``mpiexec``
     - Used command to start MPI applications.
   * - ``mpirun``
     - Alternative command for launching MPI programs.
   * - ``orterun``
     - Backend launcher for Open MPI's runtime environment.

General
=======
SmartSim provides support for two general launchers: **Dragon** and **SLURM**. Dragon is the most rapid and adaptable distributed
runtime for HPC workflows. SLURM is a open-source job scheduler and resource manager for Linux clusters.

.. list-table:: General HPC Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``srun``
     - Simple Linux Utility for Resource Management (SLURM).
   * - ``n/a``
     - High-performance computing launcher for large-scale jobs (Dragon).

Local
=====
SmartSim provides support to launch applications on the local machine.

.. list-table:: Local Launcher
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``local``
     - Runs jobs on the local machine.

The ``LaunchSettings`` class ensures that users can efficiently manage and execute their HPC jobs across
various environments.

==========
Initialize
==========
The ``LaunchSettings`` class allows you to customize your application’s execution
environment. This section details the steps to set up a ``LaunchSettings`` instance.

**Step 1: Import LaunchSettings**

After installing SmartSim, ``LaunchSettings`` may be imported in Python code like:

.. code-block:: python

    from smartsim import LaunchSettings

**Step 2: Set the Launcher Type**

Set the launcher type using either a string or a ``LauncherType`` enum. This step is important as it
determines the job's launcher. The following table lists the supported launcher strings and
their corresponding ``LauncherType`` enums:

.. list-table:: Supported Launcher Strings and Enums
   :header-rows: 1

   * - **Launcher**
     - **String**
     - **Enum**
   * - Dragon
     - ``"dragon"``
     - ``LauncherType.Dragon``
   * - SLURM
     - ``"slurm"``
     - ``LauncherType.Slurm``
   * - PALS
     - ``"pals"``
     - ``LauncherType.Pals``
   * - ALPS
     - ``"alps"``
     - ``LauncherType.Alps``
   * - Local
     - ``"local"``
     - ``LauncherType.Local``
   * - Mpiexec
     - ``"mpiexec"``
     - ``LauncherType.Mpiexec``
   * - Mpirun
     - ``"mpirun"``
     - ``LauncherType.Mpirun``
   * - Orterun
     - ``"orterun"``
     - ``LauncherType.Orterun``
   * - LSF
     - ``"lsf"``
     - ``LauncherType.Lsf``

To use an enum, import ``LauncherType`` into Python code like:

.. code-block:: python

    from smartsim import LauncherType

**Step 3: Provide Launch Arguments and Environment Variables**

Optionally, you can provide ``launch_args`` and ``env_vars``:

* ``launch_args``: A dictionary where keys are argument names (strings) and values are argument values (strings).
  These arguments are specific to the launcher being used. Example:

  .. code-block:: python

    launch_args = {"--time": "01:00:00", "--nodes": "2"}

* ``env_vars``: A dictionary where keys are environment variable names (strings) and values are environment
  variable values (strings). These variables set the environment for the application execution. Example:

  .. code-block:: python

    env_vars = {"MY_VAR": "my_value", "ANOTHER_VAR": "another_value"}

Here's how you can initialize ``LaunchSettings`` with input parameters:

**Example using a launcher String:**
Once you have imported ``LaunchSettings`` using ``from smartsim import LaunchSettings``, set the input
variable ``launcher`` to a launcher string such as `"slurm"`. For example:

.. code-block:: python

    launch_settings = LaunchSettings(
        launcher="slurm",
        launch_args={"--time": "01:00:00"},
        env_vars={"MY_VAR": "my_value"}
    )

**Example using a LauncherType Enum:**
Once you have imported ``LaunchSettings`` and ``LauncherType`` using ``from smartsim import LaunchSettings, LauncherType``,
set the input variable ``launcher`` to a ``LauncherType`` enum such as ``LauncherType.Slurm``. For example:

.. code-block:: python

    launch_settings = LaunchSettings(
        launcher=LauncherType.Slurm,
        launch_args={"--time": "01:00:00"},
        env_vars={"MY_VAR": "my_value"}
    )

======
Modify
======
After initializing a ``LaunchSettings`` object, you might want to go back and modify the ``launch_args`` or
``env_vars``. Configuring these settings allows you to change the execution environment to meet the
requirements of different jobs.

Launch Arguments
================
There are two methods to modify ``launch_args``:

1. Use ``LaunchSettings.launch_args.set``.
2. Use custom methods specific to each HPC launcher.

These functions allow you to customize launch arguments after initializing the ``LaunchSettings`` object.

**Option 1: Use LaunchSettings.launch_args.set**

To add or modify launch arguments after initializing the ``LaunchSettings`` object,
use the ``set`` method on ``launch_args`` as demonstrated below:

.. code-block:: python

    launch_settings.launch_args.set("--nodes", "2")

TODO: dragon example, syntanx is truly launcher specific

**Option 2: Use custom HPC launcher methods**

The ``LaunchSettings`` class provides custom methods to set launch arguments for different launchers.
Below are examples of how to use the custom methods for the supported launchers.

.. tabs::

   .. tab:: Dragon

      **Set Launch Arguments for Dragon:**

      .. code-block:: python

         # Set the nodes for a Dragon job
         launch_settings.launch_args.set_nodes(128)

   .. tab:: SLURM

      **Set Launch Arguments for SLURM:**

      .. code-block:: python

         # Set the nodes for a SLURM job
         launch_settings.launch_args.set_nodes(128)

   .. tab:: PALS

      **Set Launch Arguments for PALS:**

      .. code-block:: python

         # Set tasks per node for a PALS job
         launch_settings.launch_args.set_tasks_per_node(8)

   .. tab:: ALPS

      **Set Launch Arguments for ALPS:**

      .. code-block:: python

         # Set tasks per node for a ALPS job
         launch_settings.launch_args.set_tasks_per_node(8)

   .. tab:: Mpiexec

      **Set Launch Arguments for Mpiexec:**

      .. code-block:: python

         # Set tasks per node for a Mpiexec job
         launch_settings.launch_args.set_tasks_per_node(8)

   .. tab:: Mpirun

      **Set Launch Arguments for Mpirun:**

      .. code-block:: python

         # Set tasks per node for a Mpirun job
         launch_settings.launch_args.set_tasks_per_node(8)

   .. tab:: Orterun

      **Set Launch Arguments for Orterun:**

      .. code-block:: python

         # Set tasks per node for a Orterun job
         launch_settings.launch_args.set_tasks_per_node(8)

   .. tab:: LSF

      **Set Launch Arguments for LSF:**

      .. code-block:: python

         # Set tasks for a LSF job
         launch_settings.launch_args.set_tasks(8)

For detailed information on these methods, refer to the API reference page.

Environment Variables
=====================
There are two methods to configure ``env_vars``:

1. Use ``LaunchSettings.env_vars`` to overwrite the environment variables.
2. Use ``LaunchSettings.update_env`` to add environment variables.

**Option 1: Use LaunchSettings.env_vars**

To overwrite the ``env_vars`` after initializing the ``LaunchSettings`` object, set a dictionary where
each key and value are strings to ``LaunchSettings.env_vars``. This method overwrites the existing
environment settings. For example:

.. code-block:: python

    launch_settings.env_vars = {"MY_VAR": "new_value", "ANOTHER_VAR": "another_value"}

**Option 1: Use LaunchSettings.update_env**

To update the ``env_vars`` after initializing the ``LaunchSettings`` object, pass in a dictionary where
each key and value are strings to ``LaunchSettings.update_env``. This function updates the existing
environment settings without overwriting. For example:

.. code-block:: python

    launch_settings.update_env({"MY_VAR": "new_value", "ANOTHER_VAR": "another_value"})