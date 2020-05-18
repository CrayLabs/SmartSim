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

Merging
=======

When merging there are a few guidelines to follow

   - We follow the squash and merge strategy, if your commit has a number of small commits
     that do things like clean up whitespace, please squash them prior to merge.
   - Please delete the default merge message and replace with a informative merge message.
   - Wrap all merge messages to 70 characters per line.

Installation Instructions
=========================

   - Clone the git repository
      | git clone <link to repo> Smart-Sim
   - Set SmartSim env variables and add to python path
      | cd Smart-Sim && source setup_env.sh
   - Install Dependencies
      | pip install -r requirements.txt

Coding Conventions
==================

For the most part, the conventions are to follow PEP8 that is supplied by pylint. However, there
are a few things to specifically mention.

  - Underscores should preceed methods not meant to be used outside a class
  - All methods should have docstrings (with some exceptions)
  - Variable names should accurately capture what values it is storing without being overly verbose

Code Review
===========

There are multiple rounds of review that should happen for every pull request. Each
pull request will be different and require different amounts of review, but guidelines
for the review process are listed below. At each stage, the reviewer should give the
developer time to address the comments and adapt the code.

 - Address design and architectural features added. Consider future work.
 - Run test suite, comment on broken tests, test on other architectures if possible
 - Final pass, style, naming convention(could be part of design), whitespace, etc. (see Coding Conventions)
 - Approve.

Editor
======

The editor that we suggest developers to use is VSCode. Below are some extensions that
could make the process of developing on SmartSim a bit easier.

 - GitLens, for viewing changes in the git history
 - Remote SSH, for connecting to clusters and supercomputers
 - Python, Microsoft Python Language Server
 - Python indent, for correcting python indents
 - reStructuredText, for writing documentation
 - Strict Whitespace, for ensuring no whitespace left in code
 - Python Docstring Generator, for writing docstring quickly
 - C/C++, for client development
 - Settings Sync, for syncing settings accross remote servers


Documentation
=============
To make the documentation, naviagate to the base directory of SmartSim and call ``make html``.
