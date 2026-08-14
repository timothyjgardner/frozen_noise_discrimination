# Frozen-noise cortex

A small, deterministic computational model of frozen-noise discrimination.
The project tests whether accidental cochlear-channel fingerprints are strong
for short noise samples, shrink with duration, and become unreadable when
fixed internal noise is added after temporal pooling. A second experiment asks
the complementary question: whether stable differences between noise-source
statistics become easier to estimate as excerpts get longer.

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

The source-statistics extension reproduces the opposing duration trends at 20
spikes/s readout noise. Frozen-exemplar accuracy averaged over four power-law
noise families declines from 99.8% at 25 ms to 70.9% at 1.6 s. Classification
of the deliberately subtle white versus light-pink families increases from
86.8% to 99.8%, crossing the exemplar curve between 100 and 200 ms. Canonical
white-pink and pink-brown contrasts are already near ceiling at 25 ms.

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

1. Regenerates the frozen-exemplar duration-by-noise sweep.
2. Regenerates the exemplar-versus-source statistics experiment.
3. Writes CSV and dependency-free SVG outputs for both experiments.
4. Compares every reproduced field with the checked-in reference CSVs.
5. Builds the eight-page PDF report from the reproduced CSVs.
6. Reopens the PDF and verifies its page count and required sections.
7. Verifies the checked-in reference-artifact hashes.
8. Writes machine-readable reproduction metadata and SHA-256 hashes.

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
|   |-- source_statistics.py         Power-law noise families and dual tasks
|   |-- report.py                    CSV and SVG output
|   `-- cli.py                       frozen-noise-demo command
|-- scripts/
|   |-- reproduce_all.py             End-to-end deterministic reproduction
|   |-- run_source_statistics.py     Standalone source-statistics sweep
|   `-- build_report.py              Eight-page PDF report builder
|-- tests/test_model.py              Unit and integration tests
|-- results/accuracy.csv             Checked-in reference numbers
|-- results/accuracy.svg             Checked-in reference plot
|-- results/source-statistics.csv    Source/exemplar reference numbers
|-- results/source-statistics.svg    Opposing duration curves
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

## Source-statistics experiment

The extension separates two classification problems:

- **Same-source exemplar task:** train and test on noisy repetitions of two
  fixed waveforms from the same family. Only accidental finite-sample detail
  distinguishes the classes.
- **Different-source task:** train on independent waveforms from two families
  and test on new waveforms. The classifier must learn a family statistic that
  generalizes across exemplars.

The supplied families have power spectral density proportional to `1/f^a`:

- White: `a = 0`.
- Light-pink: `a = 0.5`, included as a difficult, controlled contrast.
- Pink: `a = 1`.
- Brown: `a = 2`.

All are Gaussian, band-limited to 200-7,000 Hz, and equalized for waveform RMS
by the encoder. The source decoder uses 32 independent training exemplars and
96 held-out exemplars per class; results are averaged over 12 train/test
splits. The exemplar result is averaged over eight frozen pairs from each of
the four families.

Run this experiment alone with:

```bash
python scripts/run_source_statistics.py
```

The most informative next sources are not simply farther apart on the same
spectral-slope axis. Recommended controls are:

1. **Spectrally matched amplitude-modulated noise**, varying modulation rate or
   depth while keeping the long-run carrier spectrum fixed.
2. **Notched or multiband noise**, to test localized spectral statistics rather
   than one global slope.
3. **PSD-matched non-Gaussian textures**, such as sparse filtered clicks versus
   Gaussian noise, to test envelope moments and sparsity.
4. **Blue noise (`a = -1`)**, useful as an easy high-frequency spectral anchor.

The present 16-dimensional mean-rate code can learn spectral slope. Modulation
and PSD-matched texture tests will likely require explicit envelope variance,
modulation-power, or cross-channel-correlation features.

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
  --source-csv results/source-statistics.csv \
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
