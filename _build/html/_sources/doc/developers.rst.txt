Developer
---------

Development Setup
=================

  - Fork the SmartSim repository
  - Set upstream as the main repository and set upstream push remote to ``no_push``
  - Follow installation instructions

Pull Requests
=============

Please check the following before submitting a pull request to the SmartSim repository

  1) Your feature is on a new branch off master.
  2) You are merging the feature branch from your fork into the main repository.
  3) All uncessescary whitespace has been purged from your code.
  4) All your code as been appropriately documented.
  5) The PR description is clear and concise.
  6) You have requested a review.

Installation Instructions
=========================

   - Clone the git repository
      | git clone https://github.com/Spartee/Smart-Sim.git Smart-Sim
   - Set SmartSim env variables and add to python path
      | cd Smart-Sim && source setup_env.sh
   - Install Dependencies
      | pip install -r requirements.txt
   - Install Launcher
      | cd && git clone https://stash.us.cray.com/scm/ard/poseidon-launcher.git
   - Setup Launcher
      | cd poseidon-launcher && source setenv.sh   # currently requires a patch as well
   - Install KeyDB
      | git clone https://github.com/JohnSully/KeyDB && cd KeyDB && make
      | cd src && export PATH=$PATH:$(pwd) # or add to .bashrc/.zshrc


Coding Conventions
==================

For the most part, the conventions are to follow PEP8 that is supplied by pylint. However, there
are a few things to specifically mention.

  - Underscores should preceed methods not meant to be utlized by the user in user facing classes
  - All methods should have docstrings (with some exceptions)
  - Variable names should accurately capture what values it is storing without being overly verbose


Documentation
=============
To make the documentation, naviagate to the base directory of SmartSim and call ``make html``.

