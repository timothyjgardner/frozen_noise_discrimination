PYTHON ?= python3

.PHONY: install test reproduce verify

install:
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e . --no-deps --no-build-isolation

test:
	$(PYTHON) -m unittest discover -s tests -v

reproduce:
	$(PYTHON) scripts/reproduce_all.py

verify: test reproduce
