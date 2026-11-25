# R²C-AD

A research prototype for relational and temporal anomaly detection in multivariate time series, implemented in Python with PyTorch and NumPy.

R²C-AD combines historical forecasting with cross-sensor conditional prediction to detect unusual measurements and changes in relationships between sensors. It uses robust training, optional consensus-based graph learning, empirical rank-based score fusion, and event-level alarm calibration.

## Features

- Patch-based Transformer encoder with sensor and positional embeddings.
- Temporal forecasting using historical observations only.
- Conditional prediction using historical representations and current measurements from other sensors, excluding self-edges.
- Directed sensor dependency graph learned through source ablation across chronological blocks.
- Robust median/MAD normalization and loss reweighting.
- Missing-value handling through observation masks.
- Empirical rank-based fusion of temporal and conditional residuals.
- Alarm calibration against a configurable event budget.
- Synthetic data generation and event-level evaluation.
- CPU execution by default, with optional CUDA support.

## Installation

Use a Python environment with NumPy and a recent version of PyTorch supporting `torch.load(..., weights_only=True)`.

```bash
python -m pip install numpy torch
```

The CLI uses relative imports. If you have only the supplied Python files, organize them as a package as shown below (remove any `(1)` suffixes from the downloaded filenames):

```text
project/
├── README.md
└── r2c_ad/
    ├── __init__.py
    ├── alarms.py
    ├── cli.py
    ├── data.py
    ├── learning.py
    └── model.py
```

`r2c_ad` is a suggested package name. Create an empty `__init__.py` if one is not already present. Run the following commands from the project directory. The supplied CLI defines `main()` but does not call it automatically, so the examples explicitly invoke it.

## Quick start

### 1. Generate synthetic data

```bash
python -c "from r2c_ad.cli import main; main()" demo --out demo_data
```

This generates:

| File | Description |
| --- | --- |
| `treino.csv` | 4,200 training samples |
| `teste.csv` | 1,600 test samples |
| `rotulos.csv` | Binary test anomaly labels |

The synthetic dataset contains a correlated source, a noisy copy, and a delayed signal. Anomalies are introduced by reversing the sign of the copy over selected intervals, including some intervals in the training data.

### 2. Train the detector

```bash
python -c "from r2c_ad.cli import main; main()" treinar --csv demo_data/treino.csv --model artifacts --graph consensus --budget 5
```

### 3. Detect anomalies

```bash
python -c "from r2c_ad.cli import main; main()" detectar --csv demo_data/teste.csv --model artifacts --out results.csv
```

### 4. Evaluate detection performance

```bash
python -c "from r2c_ad.cli import main; main()" avaliar --result results.csv --labels demo_data/rotulos.csv
```

The command names and output column names remain in Portuguese to match the implementation.

## Input format

Provide a CSV file with at least two uniquely named sensor columns. Each row represents one time step, in chronological order.

```csv
sensor_a,sensor_b,sensor_c
1.20,1.18,0.95
1.35,1.31,1.20
1.42,,1.35
1.50,1.47,1.42
```

- Measurements must be numeric and finite; missing values may be represented by empty fields, `NA`, or `NaN`.
- Detection data must use exactly the same sensor names and column order as training data.
- Do not include timestamp or label columns in the sensor CSV.
- Each training segment and graph block must contain more rows than the historical window length. Both the rank reference and calibration segments require at least 20 valid scored time steps.

Evaluation labels are supplied separately and must contain one binary value per input row:

```csv
anomalia
0
0
1
1
```

Here, `0` means normal and `1` means anomalous.

## Methodology

### 1. Chronological data splitting

Training data is split without randomizing temporal order:

| Partition | Share | Purpose |
| --- | ---: | --- |
| Training | 50% | Robust scaling and model optimization |
| Graph learning | 25% | Dependency estimation across blocks |
| Rank reference | 12.5% | Empirical score distributions |
| Calibration | 12.5% | Alarm threshold selection |

Each segment is windowed independently. The scaling parameters are fitted only on the training partition.

### 2. Historical representation and prediction

Historical windows are divided into patches. Measurement values and observation masks are embedded alongside learned positional and sensor embeddings. A Transformer encoder processes each sensor's historical sequence.

The model produces two predictions:

- **Temporal:** predicts the current measurement from its own historical representation.
- **Conditional:** predicts the current measurement using its historical representation and attention-based messages from other sensors, including their current values and validity indicators.

Queries and keys are derived from historical representations, and direct self-edges are excluded.

### 3. Robust optimization and graph learning

The model is optimized in two training phases using AdamW, Smooth L1 loss, source dropout, gradient clipping, and robust per-window loss weights. Missing target measurements are excluded from the loss.

In consensus mode, the first training phase is followed by source ablation across chronological graph blocks. A candidate edge must improve prediction in a majority of blocks. Eligible edges are ranked by median gain relative to its variability (MAD), and up to the configured number of incoming neighbors are retained per target. The model is then trained again with the selected graph.

| Graph mode | Behavior |
| --- | --- |
| `consensus` | Learns a sparse directed dependency graph |
| `attention` | Retains all non-self edges |
| `none` | Disables cross-sensor edges |

The learned edges represent predictive dependencies, not proven causal relationships.

### 4. Anomaly scoring

For each prediction branch, the raw score is the mean of up to three largest observed absolute residuals. The two branch scores are transformed using empirical reference ranks. Their maximum and mean are ranked again against reference distributions, and the final fused score is the larger of these two ranks.

Fully unobserved time steps receive a missing fused score.

### 5. Alarm policy and calibration

The alarm policy combines a high-score threshold, consecutive moderately high scores, and cumulative score evidence. A new alarm event begins when the alarm state changes from inactive to active.

Calibration searches 101 threshold candidates between 0.90 and 0.9999 and selects the first policy that satisfies the requested number of alarm events per 1,000 valid calibration time steps. If no candidate satisfies the budget, alarms are disabled.

The budget is an empirical calibration target, not a guaranteed future false-alarm rate.

## Training parameters

| Argument | Default | Description |
| --- | --- | --- |
| `--csv` | Required | Input CSV path |
| `--model` | Required | Model artifact directory |
| `--length` | `32` | Historical window length |
| `--patch` | `4` | Patch length; must divide the window length |
| `--width` | `32` | Transformer width; must be divisible by 4 |
| `--layers` | `1` | Number of Transformer encoder layers |
| `--epochs` | `4` | Epochs per training phase |
| `--blocks` | `5` | Graph blocks; must be at least 3 |
| `--neighbors` | `3` | Maximum incoming edges per target in consensus mode |
| `--budget` | `5` | Maximum calibration alarm events per 1,000 observed steps |
| `--graph` | `consensus` | Graph mode |
| `--seed` | `42` | PyTorch random seed |
| `--device` | `cpu` | PyTorch execution device, e.g. `cuda` |

## Generated artifacts

| File | Contents |
| --- | --- |
| `weights.pt` | Trained model state, including the graph mask |
| `reference.npz` | Scaling parameters and empirical rank references |
| `config.json` | Model settings, sensor names, threshold, and calibration statistics |
| `edges.csv` | Consensus graph audit with source, target, median gain, MAD, confidence, and selection status (header only for other graph modes) |

### Detection output

| Column | Description |
| --- | --- |
| `indice` | Zero-based input row index |
| `condicional` | Raw conditional anomaly score |
| `temporal` | Raw temporal anomaly score |
| `fusao` | Fused anomaly score |
| `ativo` | Binary alarm state |
| `inicio_alerta` | Binary alarm event onset |

The initial historical window is a warm-up period and has no anomaly scores.

## Evaluation metrics

The `avaliar` command reports JSON containing:

- `eventos_avaliaveis`: labeled anomaly events with at least one valid score.
- `eventos_sem_cobertura`: labeled events without any valid score.
- `eventos_detectados`: evaluable events containing at least one alarm onset.
- `recall_evento`: fraction of evaluable anomaly events detected.
- `falsos_alarmes`: alarm onsets during valid normal time steps.
- `exposicao_normal`: number of valid normal time steps.
- `far_por_1000`: false alarm events per 1,000 valid normal time steps.
- `atraso_mediano_amostras`: median delay from anomaly onset to first alarm onset, measured in samples.

Undefined metrics are returned as `null`.

## Source modules

| Module | Responsibility |
| --- | --- |
| `model.py` | Transformer encoder and relational prediction |
| `data.py` | CSV validation, robust scaling, and historical windows |
| `learning.py` | Training, graph learning, and residual scoring |
| `alarms.py` | Empirical rank fusion, alarm decisions, and calibration |
| `cli.py` | Data generation, training, detection, and evaluation commands |

## Limitations

- This is a research prototype, not a validated production monitoring system.
- Calibration depends on the representativeness of the reference and calibration segments.
- Robust training is intended to reduce the influence of contamination but does not guarantee robustness to arbitrary anomalies.
- Conditional prediction requires current cross-sensor observations; it is not a purely historical forecast.
- No benchmark results or performance guarantees are provided in the supplied files.
- Dependency versions and a license have not been specified in the supplied files.

