# Agent guide

## Objective

Maintain and reproduce the frozen-noise discrimination experiment. A complete
successful run must regenerate the numeric sweep, SVG plot, and PDF report,
then confirm that the numeric values match the checked-in reference CSV.

## Start here

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce_all.py
```

Do not require network access after dependency installation. Do not replace the
checked-in reference artifacts during an ordinary verification run.

## Ground truth

- Reference numbers: `results/accuracy.csv`
- Reference plot: `results/accuracy.svg`
- Reference write-up: `docs/frozen-noise-cortex-writeup.pdf`
- End-to-end runner: `scripts/reproduce_all.py`
- Model defaults: `ExperimentConfig` and `CochleaConfig`

The expected 1.6 s accuracies are 100.0%, 88.0%, 72.3%, and 59.8% at 0, 10,
20, and 40 spikes/s readout-noise SD. The clean rate distance should decline
from about 69.3 spikes/s at 25 ms to 8.2 spikes/s at 1.6 s.

## Scientific invariant

The current result depends on a whole-trace rate code. The decoder receives one
average spike rate per cochlear channel and does not receive precise spike
timing. Do not describe a duration-related loss as a general consequence of all
neural decoders. A time-resolved decoder is an explicit alternative hypothesis.

## Safe change workflow

1. Read `README.md`, `src/frozen_noise_cortex/model.py`, and
   `src/frozen_noise_cortex/experiment.py`.
2. Make the smallest scoped change.
3. Run the unit tests.
4. Run `python scripts/reproduce_all.py`.
5. If the reference numbers intentionally changed, explain why before updating
   `results/accuracy.csv`, `results/accuracy.svg`, the PDF, and
   `ARTIFACTS.sha256` together.
6. Render and inspect every PDF page after changing the report builder.

## Output policy

Ordinary reproduction writes only to `build/reproduced/`. Reference artifacts
are versioned evidence and should remain unchanged unless a model or report
revision is explicitly intended.
