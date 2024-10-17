*********
Changelog
*********

Listed here are the changes between each release of SmartSim
and SmartRedis.

Jump to :ref:`SmartRedis Changelog <changelog>`


SmartSim
========

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