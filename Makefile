PYTHON ?= .venv/bin/python

.PHONY: setup data test results

setup:            ## create the virtual environment and install the pinned packages
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

data:             ## download the BBC, RAVDESS and MovieLens data sets into data/ (214 MB, not committed)
	$(PYTHON) scripts/fetch_data.py

test:             ## run the test suite (the data set checks are skipped without data/)
	$(PYTHON) -m pytest tests -q

results:          ## regenerate every figure and number under results/
	$(PYTHON) -m experiments
