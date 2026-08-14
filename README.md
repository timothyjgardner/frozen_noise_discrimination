# Frozen-noise cortex

A small, deterministic computational model of frozen white-noise
discrimination. The project tests whether accidental cochlear-channel
fingerprints are strong for short noise samples, shrink with duration, and
become unreadable when fixed internal noise is added after temporal pooling.

The repository is self-contained. It does not require an external dataset,
web API, GPU, notebook, or proprietary software.

## Result in one paragraph

Two equal-RMS frozen white-noise samples are encoded by a 16-channel,
ERB-spaced filterbank. Upward envelope-threshold crossings generate spikes.
The decoder receives one whole-trace spike rate per channel and is trained with
regularized linear discriminant analysis. In the reference sweep, 25 ms
samples remain at or above 99.6% accuracy at every tested readout-noise level.
At 1.6 s, accuracy is 100.0%, 88.0%, 72.3%, and 59.8% for readout-noise
standard deviations of 0, 10, 20, and 40 spikes/s. Chance is 50%.

## Reproduce everything

Python 3.12 is recommended. Python 3.10 or newer is supported.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce_all.py
```

Windows PowerShell users can replace `.venv/bin/python` with
`.venv\Scripts\python.exe`.

The final command performs the complete pipeline:

1. Regenerates the Monte Carlo duration-by-noise sweep.
2. Writes a new CSV and dependency-free SVG plot.
3. Compares every reproduced numeric field with the checked-in reference CSV.
4. Builds the seven-page PDF report from the reproduced CSV.
5. Reopens the PDF and verifies its page count and required sections.
6. Verifies the checked-in reference-artifact hashes.
7. Writes machine-readable reproduction metadata and SHA-256 hashes.

Reproduced artifacts are written under `build/reproduced/`. Reference
artifacts in `results/` and `docs/` are not overwritten.

The equivalent Make targets are:

```bash
make install
make test
make reproduce
```

## Repository map

```text
.
|-- AGENTS.md                       Instructions for another coding agent
|-- README.md                       Project overview and reproduction guide
|-- requirements-lock.txt           Pinned reproduction environment
|-- pyproject.toml                   Installable Python package and CLI
|-- src/frozen_noise_cortex/
|   |-- model.py                     Filterbank and spike encoder
|   |-- experiment.py                Trial noise, LDA, and parameter sweep
|   |-- report.py                    CSV and SVG output
|   `-- cli.py                       frozen-noise-demo command
|-- scripts/
|   |-- reproduce_all.py             End-to-end deterministic reproduction
|   `-- build_report.py              Seven-page PDF report builder
|-- tests/test_model.py              Unit and integration tests
|-- results/accuracy.csv             Checked-in reference numbers
|-- results/accuracy.svg             Checked-in reference plot
|-- docs/frozen-noise-cortex-writeup.pdf
|                                     Checked-in project report
`-- .github/workflows/reproduce.yml  Clean GitHub Actions verification
```

## Model pipeline

```text
frozen waveform
    -> ERB-spaced cochlear filterbank
    -> analytic envelope per channel
    -> upward threshold-crossing spikes
    -> spike dropout plus spontaneous events
    -> whole-trace channel rates
    -> fixed Gaussian readout noise
    -> regularized two-class LDA
```

Default experiment settings:

- Durations: 25, 50, 100, 200, 400, 800, and 1,600 ms.
- Readout-noise SD: 0, 10, 20, and 40 spikes/s.
- Frozen pairs: 12 per duration.
- Trials per class and pair: 32 train and 96 test.
- Cochlear channels: 16 from 200 to 7,000 Hz.
- Spike dropout: 3%.
- Spontaneous rate: 0.2 spikes/s/channel.
- Random seed: 7.

## Run a custom sweep

After installation:

```bash
frozen-noise-demo \
  --durations-ms 25,50,100,200,400,800,1600 \
  --noise-sd 0,10,20,40 \
  --pairs 12 \
  --channels 16 \
  --seed 7 \
  --output build/custom
```

This writes `accuracy.csv` and `accuracy.svg` into the requested output
directory.

Build a report from any compatible result CSV:

```bash
python scripts/build_report.py \
  --csv build/custom/accuracy.csv \
  --output build/custom/report.pdf
```

## Interpretation boundary

This is an auditory-inspired rate-code hypothesis, not a validated model of
human cochlea or auditory cortex. The decoder discards the precise temporal
spike sequence. A time-resolved decoder could become better, rather than worse,
for longer frozen traces because it receives more identifying events. Comparing
matched rate and temporal decoders is the highest-value next experiment.

Other useful extensions are acoustic noise before the filterbank, threshold
jitter within channels, a leaky integrator with a tunable time constant, and a
more realistic auditory-nerve front end.

## Reproducibility notes

- Stimulus and trial-noise random streams are separate. Changing the list of
  noise conditions does not change the underlying frozen samples.
- The checked-in CSV is the numeric source of truth for the reference report.
- The PDF creation timestamp is expected to differ between runs. Its content,
  sections, charts, and page count are verified instead of byte identity.
- `ARTIFACTS.sha256` verifies the checked-in reference outputs before a run;
  `scripts/reproduce_all.py` performs this check automatically.
- GitHub Actions executes the tests and full reproduction on Python 3.12.
