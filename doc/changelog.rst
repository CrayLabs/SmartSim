*********
Changelog
*********

Listed here are the changes between each release of SmartSim
and SmartRedis.

Jump to :ref:`SmartRedis Changelog <changelog>`


SmartSim
========

0.4.1
-----

Released on June 24, 2022

Description:
This release of SmartSim introduces a new experimental feature to help make
SmartSim workflows more portable: the ability to run simulations models in a
container via Singularity. This feature has been tested on a small number of
platforms and we encourage users to provide feedback on its use.

We have also made improvements in a variety of areas: new utilities to load
scripts and machine learning models into the database directly from SmartSim
driver scripts and install-time choice to use either `KeyDB` or `Redis` for the
Orchestrator. The `RunSettings` API is now more consistent across subclasses. Another
key focus of this release was to aid new SmartSim users by including more
extensive tutorials and improving the documentation. The docker image containing
the SmartSim tutorials now also includes a tutorial on online training.


Launcher improvements

    - New methods for specifying `RunSettings` parameters (SmartSim-PR166_) (SmartSim-PR170_)
    - Better support for `mpirun`, `mpiexec`, and `orterun` as launchers (SmartSim-PR186_)
    - Experimental: add support for running models via Singularity (SmartSim-PR204_)

Documentation and tutorials

    - Tutorial updates (SmartSim-PR155_) (SmartSim-PR203_) (SmartSim-PR208_)
    - Add SmartSim Zoo info to documentation (SmartSim-PR175_)
    - New tutorial for demonstrating online training (SmartSim-PR176_) (SmartSim-PR188_)

General improvements and bug fixes

    - Set models and scripts at the driver level (SmartSim-PR185_)
    - Optionally use KeyDB for the orchestrator (SmartSim-PR180_)
    - Ability to specify system-level libraries (SmartSim-PR154_) (SmartSim-PR182_)
    - Fix the handling of LSF gpus_per_shard (SmartSim-PR164_)
    - Fix error when re-running `smart build`` (SmartSim-PR165_)
    - Fix generator hanging when tagged configuration variables are missing (SmartSim-PR177_)

Dependency updates

    - CMake version from 3.10 to 3.13 (SmartSim-PR152_)
    - Update click to 8.0.2 (SmartSim-PR200_)

.. _SmartSim-PR152: https://github.com/CrayLabs/SmartSim/pull/152
.. _SmartSim-PR154: https://github.com/CrayLabs/SmartSim/pull/154
.. _SmartSim-PR155: https://github.com/CrayLabs/SmartSim/pull/155
.. _SmartSim-PR164: https://github.com/CrayLabs/SmartSim/pull/164
.. _SmartSim-PR165: https://github.com/CrayLabs/SmartSim/pull/165
.. _SmartSim-PR166: https://github.com/CrayLabs/SmartSim/pull/166
.. _SmartSim-PR170: https://github.com/CrayLabs/SmartSim/pull/170
.. _SmartSim-PR175: https://github.com/CrayLabs/SmartSim/pull/175
.. _SmartSim-PR176: https://github.com/CrayLabs/SmartSim/pull/176
.. _SmartSim-PR177: https://github.com/CrayLabs/SmartSim/pull/177
.. _SmartSim-PR180: https://github.com/CrayLabs/SmartSim/pull/180
.. _SmartSim-PR182: https://github.com/CrayLabs/SmartSim/pull/182
.. _SmartSim-PR185: https://github.com/CrayLabs/SmartSim/pull/185
.. _SmartSim-PR186: https://github.com/CrayLabs/SmartSim/pull/186
.. _SmartSim-PR188: https://github.com/CrayLabs/SmartSim/pull/188
.. _SmartSim-PR200: https://github.com/CrayLabs/SmartSim/pull/200
.. _SmartSim-PR203: https://github.com/CrayLabs/SmartSim/pull/203
.. _SmartSim-PR204: https://github.com/CrayLabs/SmartSim/pull/204
.. _SmartSim-PR208: https://github.com/CrayLabs/SmartSim/pull/208

0.4.0
-----

Released on Feb 11, 2022

Description:
In this release SmartSim continues to promote ease of use.
To this end SmartSim has introduced new portability features
that allow users to abstract away their targeted hardware,
while providing even more compatibility with existing
libraries.

A new feature, Co-located orchestrator deployments has
been added which provides scalable online inference
capabilities that overcome previous performance limitations
in seperated orchestrator/application deployments.
For more information on advantages of co-located deployments,
see the Orchestrator section of the SmartSim documentation.

The SmartSim build was significantly improved to increase
customization of build toolchain and the ``smart`` command
line inferface was expanded.

Additional tweaks and upgrades have also been
made to ensure an optimal experience. Here is a
comprehensive list of changes made in SmartSim 0.4.0.


Orchestrator Enhancements:

 - Add Orchestrator Co-location (SmartSim-PR139_)
 - Add Orchestrator configuration file edit methods (SmartSim-PR109_)

Emphasize Driver Script Portability:

 - Add ability to create run settings through an experiment (SmartSim-PR110_)
 - Add ability to create batch settings through an experiment (SmartSim-PR112_)
 - Add automatic launcher detection to experiment portability functions (SmartSim-PR120_)

Expand Machine Learning Library Support:

 - Data loaders for online training in Keras/TF and Pytorch (SmartSim-PR115_) (SmartSim-PR140_)
 - ML backend versions updated with expanded support for multiple versions (SmartSim-PR122_)
 - Launch Ray internally using ``RunSettings`` (SmartSim-PR118_)
 - Add Ray cluster setup and deployment to SmartSim (SmartSim-PR50_)

Expand Launcher Setting Options:

 - Add ability to use base ``RunSettings`` on a Slurm, PBS, or Cobalt launchers (SmartSim-PR90_)
 - Add ability to use base ``RunSettings`` on LFS launcher (SmartSim-PR108_)

Deprecations and Breaking Changes

 - Orchestrator classes combined into single implementation for portability (SmartSim-PR139_)
 - ``smartsim.constants`` changed to ``smartsim.status`` (SmartSim-PR122_)
 - ``smartsim.tf`` migrated to ``smartsim.ml.tf`` (SmartSim-PR115_) (SmartSim-PR140_)
 - TOML configuration option removed in favor of environment variable approach (SmartSim-PR122_)

General Improvements and Bug Fixes:

 - Improve and extend parameter handling (SmartSim-PR107_) (SmartSim-PR119_)
 - Abstract away non-user facing implementation details (SmartSim-PR122_)
 - Add various dimensions to the CI build matrix for SmartSim testing (SmartSim-PR130_)
 - Add missing functions to LSFSettings API (SmartSim-PR113_)
 - Add RedisAI checker for installed backends (SmartSim-PR137_)
 - Remove heavy and unnecessary dependencies (SmartSim-PR116_) (SmartSim-PR132_)
 - Fix LSFLauncher and LSFOrchestrator (SmartSim-PR86_)
 - Fix over greedy Workload Manager Parsers (SmartSim-PR95_)
 - Fix Slurm handling of comma-separated env vars (SmartSim-PR104_)
 - Fix internal method calls (SmartSim-PR138_)

Documentation Updates:

 - Updates to documentation build process (SmartSim-PR133_) (SmartSim-PR143_)
 - Updates to documentation content (SmartSim-PR96_) (SmartSim-PR129_) (SmartSim-PR136_) (SmartSim-PR141_)
 - Update SmartSim Examples (SmartSim-PR68_) (SmartSim-PR100_)


.. _SmartSim-PR50: https://github.com/CrayLabs/SmartSim/pull/50
.. _SmartSim-PR68: https://github.com/CrayLabs/SmartSim/pull/68
.. _SmartSim-PR86: https://github.com/CrayLabs/SmartSim/pull/86
.. _SmartSim-PR90: https://github.com/CrayLabs/SmartSim/pull/90
.. _SmartSim-PR95: https://github.com/CrayLabs/SmartSim/pull/95
.. _SmartSim-PR96: https://github.com/CrayLabs/SmartSim/pull/96
.. _SmartSim-PR100: https://github.com/CrayLabs/SmartSim/pull/100
.. _SmartSim-PR104: https://github.com/CrayLabs/SmartSim/pull/104
.. _SmartSim-PR107: https://github.com/CrayLabs/SmartSim/pull/107
.. _SmartSim-PR108: https://github.com/CrayLabs/SmartSim/pull/108
.. _SmartSim-PR109: https://github.com/CrayLabs/SmartSim/pull/109
.. _SmartSim-PR110: https://github.com/CrayLabs/SmartSim/pull/110
.. _SmartSim-PR112: https://github.com/CrayLabs/SmartSim/pull/112
.. _SmartSim-PR113: https://github.com/CrayLabs/SmartSim/pull/113
.. _SmartSim-PR115: https://github.com/CrayLabs/SmartSim/pull/115
.. _SmartSim-PR116: https://github.com/CrayLabs/SmartSim/pull/116
.. _SmartSim-PR118: https://github.com/CrayLabs/SmartSim/pull/118
.. _SmartSim-PR119: https://github.com/CrayLabs/SmartSim/pull/119
.. _SmartSim-PR120: https://github.com/CrayLabs/SmartSim/pull/120
.. _SmartSim-PR122: https://github.com/CrayLabs/SmartSim/pull/122
.. _SmartSim-PR129: https://github.com/CrayLabs/SmartSim/pull/129
.. _SmartSim-PR130: https://github.com/CrayLabs/SmartSim/pull/130
.. _SmartSim-PR132: https://github.com/CrayLabs/SmartSim/pull/132
.. _SmartSim-PR133: https://github.com/CrayLabs/SmartSim/pull/133
.. _SmartSim-PR136: https://github.com/CrayLabs/SmartSim/pull/136
.. _SmartSim-PR137: https://github.com/CrayLabs/SmartSim/pull/137
.. _SmartSim-PR138: https://github.com/CrayLabs/SmartSim/pull/138
.. _SmartSim-PR139: https://github.com/CrayLabs/SmartSim/pull/139
.. _SmartSim-PR140: https://github.com/CrayLabs/SmartSim/pull/140
.. _SmartSim-PR141: https://github.com/CrayLabs/SmartSim/pull/141
.. _SmartSim-PR143: https://github.com/CrayLabs/SmartSim/pull/143


0.3.2
-----

Released on August 10, 2021

Description:

 - Upgraded RedisAI backend to 1.2.3 (SmartSim-PR69_)
 - PyTorch 1.7.1, TF 2.4.2, and ONNX 1.6-7 (SmartSim-PR69_)
 - LSF launcher for IBM machines (SmartSim-PR62_)
 - Improved code coverage by adding more unit tests (SmartSim-PR53_)
 - Orchestrator methods to get address and check status (SmartSim-PR60_)
 - Added Manifest object that tracks deployables in Experiments (SmartSim-PR61_)
 - Bug fixes (SmartSim-PR52_) (SmartSim-PR58_) (SmartSim-PR67_) (SmartSim-PR73_)
 - Updated documentation and examples (SmartSim-PR51_) (SmartSim-PR57_) (SmartSim-PR71_)
 - Improved IP address aquisition (SmartSim-PR72_)
 - Binding database to network interfaces

.. _SmartSim-PR51: https://github.com/CrayLabs/SmartSim/pull/51
.. _SmartSim-PR52: https://github.com/CrayLabs/SmartSim/pull/52
.. _SmartSim-PR53: https://github.com/CrayLabs/SmartSim/pull/53
.. _SmartSim-PR57: https://github.com/CrayLabs/SmartSim/pull/57
.. _SmartSim-PR58: https://github.com/CrayLabs/SmartSim/pull/58
.. _SmartSim-PR60: https://github.com/CrayLabs/SmartSim/pull/60
.. _SmartSim-PR61: https://github.com/CrayLabs/SmartSim/pull/61
.. _SmartSim-PR62: https://github.com/CrayLabs/SmartSim/pull/62
.. _SmartSim-PR67: https://github.com/CrayLabs/SmartSim/pull/67
.. _SmartSim-PR69: https://github.com/CrayLabs/SmartSim/pull/69
.. _SmartSim-PR71: https://github.com/CrayLabs/SmartSim/pull/71
.. _SmartSim-PR72: https://github.com/CrayLabs/SmartSim/pull/72
.. _SmartSim-PR73: https://github.com/CrayLabs/SmartSim/pull/73

0.3.1
-----

Released on May 5, 2021

Description:
This release was dedicated to making the install process
easier. SmartSim can be installed from PyPi now and the
``smart`` cli tool makes installing the machine learning
runtimes much easier.

 - Pip install (SmartSim-PR42_)
 - ``smart`` cli tool for ML backends (SmartSim-PR42_)
 - Build Documentation for updated install (SmartSim-PR43_)
 - Migrate from Jenkins to Github Actions CI (SmartSim-PR42_)
 - Bug fix for setup.cfg (SmartSim-PR35_)

.. _SmartSim-PR43: https://github.com/CrayLabs/SmartSim/pull/43
.. _SmartSim-PR42: https://github.com/CrayLabs/SmartSim/pull/42
.. _SmartSim-PR35: https://github.com/CrayLabs/SmartSim/pull/35

0.3.0
-----

Released on April 1, 2021

Description:

 - initial 0.3.0 (first public) release of SmartSim


---------------------------------------------------------------

SmartRedis
==========

.. _changelog:

.. include:: ../smartredis/doc/changelog.rst
    :start-line: 3