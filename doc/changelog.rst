*********
Changelog
*********

Listed here are the changes between each release of SmartSim
and SmartRedis.

Jump to :ref:`SmartRedis Changelog <changelog>`


SmartSim
========

0.4.0
-----

Expected release on Feb 11, 2022

Description:

 - Add ability to create run settings through an experiment (SmartSim-PR110_)
 - Add ability to create batch settings through an experiment (SmartSim-PR112_)
 - Add automatic launcher detection to experiment poratability functions (SmartSim-PR120_)
 - Add Ray cluster setup and deployment to SmartSim (SmartSim-PR50_)
 - Add machine learning data loaders for Keras/TF and Pytorch (SmartSim-PR115_)
 - Improve and extend parameter handling (SmartSim-PR107_) (SmartSim-PR119_)
 - Abstract non-user facing implimentation details (SmartSim-PR122_)
 - Add various dimensions to the CI build matrix (SmartSim-PR130_)
 - Remove heavy and unnecessary dependencies (SmartSim-PR116_) (SmartSim-PR132_)
 - Bug fixes (SmartSim-PR95_) (SmartSim-PR104_) (SmartSim-PR138_)
 - Update SmartSim Examples (SmartSim-PR68_) (SmartSim-PR100_)
 - Updates to documentation build process (SmartSim-PR133_)
 - Updates to documentation content (SmartSim-PR96_) (SmartSim-PR129_)

 - TODO: Not Sure (SmartSim-PR86_) (SmartSim-PR90_) (SmartSim-PR108_) (SmartSim-PR113_) (SmartSim-PR118_)

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
.. _SmartSim-PR138: https://github.com/CrayLabs/SmartSim/pull/138


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