**************
LaunchSettings
**************
========
Overview
========
The ``LaunchSettings`` class manages launcher configuration settings to enable the injection of
launcher-specific behavior into jobs. SmartSim ``LaunchSettings`` supports several launchers and allows users to
configure launch arguments and environment variables. Additionally, the object provides methods to access and modify
launch arguments, environment variables and retrieve information associated with the ``LaunchSettings`` object.

**Dragon** is the fastest and most versatile distributed runtime available for HPC workflows. It can function as a
launcher within a Slurm allocation, providing rapid, interactive, and customizable execution of complex workflows
on large HPC systems. As a scheduler-agnostic solution, Dragon allows the same SmartSim script to run seamlessly
on both Slurm and PBS systems, with future support for additional schedulers.

===================
Supported Launchers
===================
The ``LaunchSettings`` class supports multiple launchers, each customized for different environments. Among these,
**Dragon** stands out as the fastest and most versatile option, ideal for large-scale HPC jobs. The following launchers
are categorized based on their specific use cases and the systems they are designed for.

HPE Cray Specific Launchers
===========================
SLURM and PALS launchers are specifically designed for HPE Cray systems, providing efficient management and
execution of parallel applications.

.. list-table:: HPE Cray Specific Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``aprun``
     - Application Level Placement Scheduler for HPE Cray systems.
   * - ``palsrun``
     - Parallel Application Launch Service for distributed computing.


IMB Specific Launchers
======================
The LSF launcher is tailored for use with the IBM Spectrum Load Sharing Facility (LSF), which is commonly used in
enterprise environments to distribute and manage workloads.

.. list-table:: IMB Specific Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``jsrun``
     - Load Sharing Facility for launching parallel jobs.

Standard MPI Launchers
======================
These launchers are used for running MPI (Message Passing Interface) applications, providing flexibility
and control over parallel job execution.

.. list-table:: MPI Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``mpiexec``
     - Standard MPI launcher for parallel jobs.
   * - ``mpirun``
     - Another MPI launcher, often used with Open MPI.
   * - ``orterun``
     - Open MPI's runtime environment launcher.

General HPC Launchers
=====================
The SLURM and Dragon launchers are suitable for general high-performance computing (HPC) environments, offering robust solutions
for job scheduling and resource management.

.. list-table:: General HPC Launchers
   :header-rows: 1

   * - **Run Command**
     - **Description**
   * - ``srun``
     - Open-source job scheduler and resource manager for Linux clusters.
   * - ``dragonrun``
     - High-performance computing launcher for large-scale jobs.

Other
=====

.. list-table:: General HPC Launchers
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
The ``LaunchSettings`` class provides a way to configure and manage the execution environment for your
applications. This section outlines the steps to configure the essential parameters that determine how
your application launches.

**Step 1: Import LaunchSettings**

After installing Smartsim, ``LaunchSettings`` may be imported in Python code like:

.. code-block:: python

    from smartsim import LaunchSettings

**Step 2: Set the Launcher Type**

Set the launcher type using either a string or a ``LauncherType`` enum. This step is crucial as it
determines the specific launcher configuration that will be applied. The following table lists the
supported strings and their corresponding enums:

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


**Step 3: Provide Launch Arguments and Environment Variables**
Optionally, you can provide ``launch_args`` and ``env_vars`` to customize the job execution environment:

* ``launch_args``: A dictionary where keys are argument names (strings) and values are argument values (strings).
  These arguments are specific to the launcher being used. Example:

  .. code-block:: python

    launch_args = {"--time": "01:00:00", "--nodes": "2"}

* ``env_vars``: A dictionary where keys are environment variable names (strings) and values are environment
  variable values (strings). These variables set the environment for the job execution. Example:

  .. code-block:: python

    env_vars = {"MY_VAR": "my_value", "ANOTHER_VAR": "another_value"}

Here's how you can initialize ``LaunchSettings`` with these parameters:

**Example using a launcher String:**
Once you have imported ``LaunchSettings`` using ``from smartsim import LaunchSettings``, use a
launcher string such as `"slurm"`. For example:

.. code-block:: python

    launch_settings = LaunchSettings(
        launcher="slurm",
        launch_args={"--time": "01:00:00"},
        env_vars={"MY_VAR": "my_value"}
    )

**Example using an Enum:**
Once you have imported ``LaunchSettings`` using ``from smartsim import LaunchSettings``, use a
``LauncherType`` enum such as ``LauncherType.Slurm``. For example:

.. code-block:: python

    launch_settings = LaunchSettings(
        launcher=LauncherType.Slurm,
        launch_args={"--time": "01:00:00"},
        env_vars={"MY_VAR": "my_value"}
    )

=========
Configure
=========
After initializing a ``LaunchSettings`` object, you might want to go back and configure ``launch_args`` or
``env_vars`` for several reasons. Customizing these settings allows you to fine-tune the execution environment
to meet the specific needs of different jobs, optimize performance, and achieve better resource utilization and
faster execution times. Additionally, modifying these settings can aid in debugging by enabling additional
logging, setting debug flags, or changing the execution environment. Experimentation with different configurations
can also help you understand how changes affect performance or behavior, which is particularly useful in research
and development settings. Finally, adjusting these settings allows for a more personalized and efficient workflow,
accommodating different user preferences and requirements.

Launch Arguments
================
Configure `launch_args` in two ways:

1. Use ``LaunchSettings.launch_args.set``.
2. Use custom methods specific to each launcher.

These functions allow you to customize `launch_args` after initializing the ``LaunchSettings`` object.

**Option 1: Use LaunchSettings.launch_args.set**

To set additional launch arguments after initializing the ``LaunchSettings`` object,
use the `set` method on `launch_args` as shown below:

.. code-block:: python

    launch_settings.launch_args.set("--nodes", "2")

**Option 2: Use custom methods specific to each launcher**

The ``LaunchSettings`` class provides custom methods to set launch arguments for different supported launchers.
These methods tailor the job execution environment to meet specific requirements of each launcher.
Below are examples of how to use these custom methods for various launchers.

.. tabs::

   .. tab:: Dragon

      **Set Launch Arguments for Dragon:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: SLURM

      **Set Launch Arguments for SLURM:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: PALS

      **Set Launch Arguments for PALS:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: ALPS

      **Set Launch Arguments for ALPS:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: Local

      **Set Launch Arguments for Local:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: Mpiexec

      **Set Launch Arguments for Mpiexec:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: Mpirun

      **Set Launch Arguments for Mpirun:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: Orterun

      **Set Launch Arguments for Orterun:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

   .. tab:: LSF

      **Set Launch Arguments for LSF:**

      .. code-block:: python

         launch_settings.launch_args.set("--nodes", "2")

For detailed information on these methods, refer to the API reference page.

Environment Variables
=====================
To update the ``env_vars`` after initializing the ``LaunchSettings`` object, pass in a dictionary where
each key and value are strings to ``LaunchSettings.update_env``. This function updates the existing dictionary
of environment settings without overwriting it. For example:

.. code-block:: python

    launch_settings.update_env({"MY_VAR": "new_value", "ANOTHER_VAR": "another_value"})

========
Examples
========

Local
=====

HPC
===