
MAKEFLAGS += --no-print-directory

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: SmartSim Makefile help
# help:

ifndef COV_FILE
export COV_FILE="${PWD}/tests/test_configs/cov/local_cov.cfg"
endif

SHELL:=/bin/bash

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help:
# help: Build
# help: -------

# help: silc                           - Make SILC python client and setup SILC library
.PHONY: silc
silc:
	@bash ./build-scripts/build-silc.sh

# help: deps                           - Make SmartSim dependencies (CPU builds, TF and PT)
.PHONY: deps
deps:
	@bash ./build-scripts/build-deps.sh

# help: deps-cpu-all                   - Make SmartSim dependencies (CPU builds, TF, PT, TFL, ONNX)
.PHONY: deps-cpu-all
deps-cpu-all:
	@bash ./build-scripts/build-deps.sh cpu 1 1 1 1

# help: deps-gpu                       - Make SmartSim dependencies (GPU builds, TF, PT)
.PHONY: deps-gpu
deps-gpu:
	@bash ./build-scripts/build-deps.sh gpu

# help: deps-gpu-all                   - Make SmartSim dependencies (GPU builds, TF, PT, TFL, ONNX)
.PHONY: deps-gpu-all
deps-gpu-all:
	@bash ./build-scripts/build-deps.sh gpu 1 1 1 1


# help: clean-deps                     - remove third-party deps
.PHONY: clean-deps
clean-deps:
	@rm -rf ./third-party


# help: clean                          - remove builds, pyc files, .gitignore rules
.PHONY: clean
clean:
	@git clean -X -f -d


# help: clobber                        - clean, remove deps, builds, (be careful)
.PHONY: clobber
clobber: clean clean-deps


# help:
# help: Style
# help: -------

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


# help:
# help: Documentation
# help: -------

# help: docs                           - generate project documentation
.PHONY: docs
docs:
	@cd doc; make html


# help:
# help: Test
# help: -------

# help: test                           - Build and run all tests (C, C++, Fortran, Python)
.PHONY: test
test:
	@cd ./tests/; python -m pytest

# help: test-verbose                   - Build and run all tests [verbosely]
.PHONY: test-verbose
test-verbose:
	@cd ./tests/; python -m pytest -vv

# help: test-cov                       - run python tests with coverage
.PHONY: test-cov
test-cov:
	@cd ./tests/; python -m pytest --cov=../smartsim -vv --cov-config=${COV_FILE}

