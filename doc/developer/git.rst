
************
Git Workflow
************

Setup
=====

  - Fork the SmartSim repository
  - Set upstream as the main repository and set upstream push remote to ``no_push``
  - Follow installation instructions

Pull Requests
=============

Please check the following before submitting a pull request to the SmartSim repository

  1) Your feature is on a new branch off master.
  2) You are merging the feature branch from your fork into the main repository.
  3) All unnecessary whitespace has been purged from your code.
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

