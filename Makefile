# Makefile for stratified-turbulence-closures

PYTHON := .venv/bin/python3
ifeq (,$(wildcard $(PYTHON)))
PYTHON := python3
endif

.PHONY: all verify figures controls numbers clean

all: verify

verify:
	@echo "=== Running numerical verification ==="
	$(PYTHON) verify_numbers.py
	@echo "=== Verification complete ==="

numbers:
	@echo "=== Rebuilding paper/numbers.json ==="
	$(PYTHON) src/build_numbers_json.py

figures:
	@echo "=== Regenerating figures ==="
	$(PYTHON) src/generate_figure_edition2.py

controls:
	@echo "=== Re-running controls ==="
	$(PYTHON) src/round2_controls.py
	$(PYTHON) src/round3_investigations.py

clean:
	rm -rf .cache/matplotlib
