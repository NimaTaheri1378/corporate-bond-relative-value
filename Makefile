SHELL := /bin/bash
N_JOBS ?= $(shell python -c "import os; print(os.getenv('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)")
export RV_N_JOBS := $(N_JOBS)
export POLARS_MAX_THREADS := $(N_JOBS)
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export NUMEXPR_NUM_THREADS ?= 1
export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)

.PHONY: env test compile discover demo-visual docs tree clean-logs

env:
	python scripts/check_environment.py

compile:
	python -m compileall -q src scripts

test: compile
	python -m pytest -q

discover:
	python scripts/discover_wrds.py --config configs/data.yaml --output-dir data/manifests/wrds_schema --describe-level likely --max-describe-per-library 300

demo-visual:
	python scripts/build_demo_visual.py

docs:
	mkdocs build --strict

tree:
	@if command -v tree >/dev/null 2>&1; then tree -a -L 4 -I '.git|__pycache__|*.pyc|run_logs'; else find . -maxdepth 4 -not -path './.git/*' -not -path './run_logs/*' | sort; fi

clean-logs:
	find run_logs -type f -name '*.log' -delete
.PHONY: validate-contracts profile-tables smoke-pull pull-static pull-monthly

validate-contracts:
	python scripts/validate_step02_contracts.py

profile-tables:
	python scripts/profile_wrds_tables.py --make-figures

smoke-pull:
	python scripts/pull_wrds.py --scope smoke --smoke-limit 1000 --workers 2

pull-static:
	python scripts/pull_wrds.py --scope static --workers 4

pull-monthly:
	python scripts/pull_wrds.py --scope monthly_returns --workers 4
# --- Step 03 extraction targets ---
.PHONY: step03-plan step03-core step03-trace-pilot step03-trace-full inspect-raw

step03-plan:
	python scripts/run_step03_extract.py --phase plan --output-root data/raw/wrds/v1

step03-core:
	python scripts/run_step03_extract.py --phase core --output-root data/raw/wrds/v1 --workers $${WRDS_PARALLEL_WORKERS:-4}

step03-trace-pilot:
	python scripts/run_step03_extract.py --phase trace --output-root data/raw/wrds/v1 --workers $${TRACE_WRDS_WORKERS:-2} --trace-pilot-months $${TRACE_PILOT_MONTHS:-1}

step03-trace-full:
	python scripts/run_step03_extract.py --phase trace --output-root data/raw/wrds/v1 --workers $${TRACE_WRDS_WORKERS:-2} --trace-start-date 2002-07-01 --trace-end-date 2025-04-01

inspect-raw:
	python scripts/inspect_raw_lake.py --manifest-glob "data/manifests/extractions/*.csv" --output-dir data/manifests/extractions/raw_lake_report --make-figures
# --- Step 03b robust extraction targets ---
.PHONY: step03b-plan step03b-core-resume step03b-trace-pilot step03b-inspect

step03b-plan:
	python scripts/run_step03_extract.py --phase plan --output-root data/raw/wrds/v1 --core-window quarter --trace-window month

step03b-core-resume:
	python scripts/run_step03_extract.py --phase core --output-root data/raw/wrds/v1 --workers $${WRDS_PARALLEL_WORKERS:-4} --engine subprocess --core-window quarter

step03b-trace-pilot:
	python scripts/run_step03_extract.py --phase trace --output-root data/raw/wrds/v1 --workers $${TRACE_WRDS_WORKERS:-3} --engine subprocess --trace-window month --trace-pilot-months $${TRACE_PILOT_MONTHS:-6}

step03b-inspect:
	python scripts/inspect_raw_lake.py --manifest-glob "data/manifests/extractions/step03*.csv" --output-dir data/manifests/extractions/raw_lake_report --make-figures
