# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


MAKEFLAGS += --no-print-directory

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: SmartSim Makefile help
# help:

# If COV_FILE is an empty string, no file will be used (the whole
# source code will be considered reachable by coverage)
# If COV_FILE is not defined, only local launcher code will be
# checked for coverage
ifndef COV_FILE
export COV_FILE="${PWD}/tests/test_configs/cov/local_cov.cfg"
endif

SHELL:=/bin/bash

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help:
# help: Clean
# help: -----

# help: clean                          - remove builds, pyc files, .gitignore rules
.PHONY: clean
clean:
	@git clean -X -f -d


# help: clobber                        - clean, remove deps, builds, (be careful)
.PHONY: clobber
clobber: clean


# help:
# help: Style
# help: -------
# help: check-all                      - Performs all the style-related checks
.PHONY: check-all
check-all: check-style check-format check-sort-imports check-lint check-mypy
	$(info All style checks PASSED)

# help: style                          - Sort imports and format with black
.PHONY: style
style: sort-imports format


# help: check-style                    - check code style compliance
.PHONY: check-style
check-style: check-sort-imports check-format


# help: format                         - perform code style format
.PHONY: format
format:
	@black ./smartsim ./tests/


# help: check-format                   - check code format compliance
.PHONY: check-format
check-format:
	@black --check ./smartsim ./tests/


# help: sort-imports                   - apply import sort ordering
.PHONY: sort-imports
sort-imports:
	@isort ./smartsim ./tests/ --profile black


# help: check-sort-imports             - check imports are sorted
.PHONY: check-sort-imports
check-sort-imports:
	@isort ./smartsim ./tests/ --check-only --profile black


# help: check-lint                     - run static analysis checks
.PHONY: check-lint
check-lint:
	@pylint --rcfile=.pylintrc ./smartsim


# help: check-mypy                     - run static type check
.PHONY: check-mypy
check-mypy:
	@mypy --config-file=./pyproject.toml


# help:
# help: Documentation
# help: -------

# help: docs                           - generate project documentation
.PHONY: docs
docs:
	@cd doc; make html

# help: docks                          - generate project documentation with docker
.PHONY: docks
docks:
	@rm -rf docs/develop
	@mkdir -p docs
	@docker compose build --progress=plain docs-dev
	@docker create -ti --name devdocs smartsim-docs:dev-latest
	@docker cp devdocs:/usr/local/src/SmartSim/doc/_build/html/ ./docs/develop
	@docker container rm devdocs
	@cp -r .docs_static/. ./docs/

# help: cov                            - generate html coverage report for Python client
.PHONY: cov
cov:
	@coverage html
	@echo if data was present, coverage report is in ./htmlcov/index.html


# help: tutorials-dev                  - Build and start a docker container to run the tutorials
.PHONY: tutorials-dev
tutorials-dev:
	@docker compose build tutorials-dev
	@docker run -p 8888:8888 smartsim-tutorials:dev-latest

# help: tutorials-prod                 - Build and start a docker container to run the tutorials (v0.8.0)
.PHONY: tutorials-prod
tutorials-prod:
	@docker compose build tutorials-prod
	@docker run -p 8888:8888 smartsim-tutorials:v0.8.0


# help:
# help: Test
# help: -------

# help: test                           - Run all tests
.PHONY: test
test:
	@python -m pytest --ignore=tests/full_wlm/ --ignore=tests/dragon

# help: test-verbose                   - Run all tests verbosely
.PHONY: test-verbose
test-verbose:
	@python -m pytest -vv --ignore=tests/full_wlm/ --ignore=tests/dragon

# help: test-debug                     - Run all tests with debug output
.PHONY: test-debug
test-debug:
	@SMARTSIM_LOG_LEVEL=developer python -m pytest -s -o log_cli=true -vv --ignore=tests/full_wlm/ --ignore=tests/dragon

# help: test-cov                       - Run all tests with coverage
.PHONY: test-cov
test-cov:
	@python -m pytest -vv --cov=./smartsim --cov-config=${COV_FILE} --ignore=tests/full_wlm/ --ignore=tests/dragon


# help: test-full                      - Run all WLM tests with Python coverage (full test suite)
# help:                                  WARNING: do not run test-full on shared systems.
.PHONY: test-full
test-full:
	@python -m pytest --cov=./smartsim -vv --cov-config=${COV_FILE}

# help: test-wlm                   - Run the wlm-specific tests
.PHONY: test-wlm
test-wlm:
	@python -m pytest -vv tests/full_wlm/ tests/on_wlm

# help: test-dragon                   - Run dragon-specific tests
.PHONY: test-dragon
test-dragon:
	@dragon pytest tests/dragon
