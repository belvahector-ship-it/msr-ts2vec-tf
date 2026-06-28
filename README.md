# Multi-Resolution Temporal Encoding for Self-Supervised Cryptocurrency Market State Discovery

> **A Controlled Empirical Study**
>
> Belva Fahrozi Chiangmaitri (P31202502702) — MTI_48, Universitas Dian Nuswantoro
> 
> *Status: Active Development — Implementation Phase*

---

## Overview

This repository implements a controlled empirical study investigating whether static
multi-resolution temporal input improves the quality of self-supervised latent
representations for cryptocurrency market state discovery.

The encoder backbone is **TS2Vec** (Yue et al., AAAI 2022), operated in a
**Multi-Branch Fixed-Dimension Late Fusion** architecture: four independent TS2Vec
branches (15m, 1h, 4h, 1d) each produce 64-dim embeddings, which are concatenated
and projected to a fixed 256-dim representation via a frozen random projection matrix.
Market states are discovered via **HDBSCAN** clustering on the 256-dim space.

The framing is a **controlled empirical study** — temporal resolution is the sole
independent variable. All other factors (architecture, hyperparameters, normalization,
clustering, evaluation protocol, seeds) are held identical across all conditions.

---

## Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | Does static multi-resolution temporal input improve self-supervised market state representation quality vs. single resolution? |
| RQ2 | How does the number of active temporal resolutions influence representation quality and stability? |
| RQ3 | Do discovered market states exhibit statistically distinguishable return distributions consistent with known market regimes? |

---

## Architecture

```
Branch 15m: windows [N, 48, 7] → TS2Vec → max-pool → e_15m [N, 64]  ─┐
Branch 1h:  windows [N, 48, 7] → TS2Vec → max-pool → e_1h  [N, 64]  ─┤
Branch 4h:  windows [N, 48, 7] → TS2Vec → max-pool → e_4h  [N, 64]  ─┤→ concat → fixed random proj → [N, 256] → HDBSCAN
Branch 1d:  windows [N, 48, 7] → TS2Vec → max-pool → e_1d  [N, 64]  ─┘

All conditions (1TF–4TF) produce identical 256-dim embeddings.
No learnable parameters in the fusion step.
```

---

## Experimental Conditions

| Label | Active Branches | Concat Dim | Output Dim |
|-------|----------------|------------|------------|
| 1TF   | {1h}           | 64         | 256        |
| 2TF   | {15m, 1h}      | 128        | 256        |
| 3TF   | {15m, 1h, 4h}  | 192        | 256        |
| 4TF   | {15m, 1h, 4h, 1d} | 256    | 256        |
| BL-15m | {15m}         | 64         | 256        |
| BL-4h  | {4h}          | 64         | 256        |
| BL-1d  | {1d}          | 64         | 256        |

**Total runs:** 8 conditions × 5 seeds = 40 experiment runs.

---

## Dataset

| Property | Value |
|----------|-------|
| Asset    | BTC/USDT |
| Exchange | Binance |
| Period   | 2020-01-01 – 2023-12-31 UTC |
| Timeframes | 15m, 1h, 4h, 1d |
| Train split | 2020-01-01 – 2022-12-31 (walk-forward) |
| Test split  | 2023-01-01 – 2023-12-31 |
| Features | 7 OHLCV-derived (no technical indicators) |
| Window size | W = 48 candles, stride = 1 |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Git
- (Optional but recommended) NVIDIA GPU with CUDA

### 2. Clone and Install

```bash
git clone https://github.com/{AUTHOR_GITHUB}/crypto-ssl-market-states.git
cd crypto-ssl-market-states

# Create environment
conda env create -f environment.yml
conda activate crypto-ssl

# Or with pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m pytest tests/test_utils/ -v
```

### 4. Download Data

```bash
python scripts/download_data.py --config configs/base.yaml
```

This downloads BTC/USDT OHLCV data for all four timeframes and saves to `data/raw/`.
Expected download time: 5–15 minutes depending on connection and Binance rate limits.

### 5. Run Full Pipeline

```bash
# Run one condition and seed (for testing)
python scripts/run_experiment.py --condition 1tf --seed 42

# Run all 40 experiments
python scripts/run_experiment.py --run-all
```

### 6. Generate Paper Outputs

```bash
python scripts/generate_artifacts.py
```

Outputs appear in `outputs/final/`.

---

## Repository Structure

```
crypto-ssl-market-states/
├── configs/                    # YAML configuration files
│   ├── base.yaml               # All shared hyperparameters (single source of truth)
│   └── experiment_*.yaml       # Per-condition overrides
├── data/                       # NOT committed — see download instructions above
│   ├── raw/                    # Raw Parquet files from Binance API
│   ├── interim/                # Aligned master DataFrame
│   └── processed/              # Features, windows, train/test splits
├── src/                        # Source modules
│   ├── acquisition/            # M1: Binance data download
│   ├── preprocessing/          # M3: temporal alignment
│   ├── features/               # M4: feature engineering
│   ├── datasets/               # M6: window generation + PyTorch Dataset
│   ├── models/                 # M7: TS2Vec wrapper + fusion module
│   ├── trainers/               # M8: branch training orchestration
│   ├── evaluation/             # M11: geometric + economic metrics
│   ├── clustering/             # M10: HDBSCAN two-stage protocol
│   ├── visualization/          # M12: publication figures
│   └── utils/                  # M0: config, logging, seed, device, paths
├── scripts/                    # Runnable entry points
├── tests/                      # Unit and integration tests
├── experiments/                # NOT committed — runtime experiment artifacts
├── checkpoints/                # NOT committed — uploaded to Zenodo separately
├── outputs/                    # NOT committed — generated paper artifacts
├── docs/                       # Design and implementation documents
│   ├── DS-01_Architecture_Decision_Records.md
│   ├── DS-02_Data_Flow_Specification.md
│   ├── DS-03_Research_Protocol_Specification.md
│   ├── DS-04_Testing_and_Validation_Specification.md
│   └── IMP-01_Implementation_Roadmap.md
├── logs/                       # NOT committed — runtime logs
├── requirements.txt
├── environment.yml
└── README.md
```

---

## TS2Vec Dependency

This project uses TS2Vec (Yue et al., AAAI 2022) as an external pinned dependency.
The original source is never modified.

```
Pinned commit: {TS2VEC_COMMIT_HASH}
Primary source: https://github.com/yuezhihan/ts2vec
Fallback fork:  https://github.com/{AUTHOR_GITHUB}/ts2vec  (same commit)
```

> **Action required at project start:** Fork the TS2Vec repository and record the
> commit hash in this README, `requirements.txt`, and `configs/base.yaml`.
> See IMP-01 Risk Register R-01.

---

## Reproducibility

All results are deterministic given the same seeds and configuration.

**Five seeds used:** `[42, 123, 456, 789, 1024]`

**Checkpoint archive:** [Link to Zenodo — to be added post-experiment]

To reproduce results from the checkpoint archive:
```bash
# Download checkpoints from Zenodo
python scripts/download_checkpoints.py --zenodo-doi {DOI}

# Evaluate without retraining
python scripts/run_experiment.py --eval-only --run-all
```

---

## Design Documents

All research decisions are documented in `docs/`:

| Document | Contents |
|----------|----------|
| DS-01 | Architecture Decision Records (ADRs 001–020) |
| DS-02 | Data Flow Specification (Stages 0–9) |
| DS-03 | Research Protocol Specification |
| DS-04 | Testing and Validation Specification |
| IMP-01 | Implementation Roadmap (M0–M15) |

Research decisions are frozen. Any deviation is documented as a formal ADR update.

---

## Citation

```bibtex
@article{chiangmaitri2026multiresolution,
  title   = {Multi-Resolution Temporal Encoding for Self-Supervised Cryptocurrency
             Market State Discovery: A Controlled Empirical Study},
  author  = {Chiangmaitri, Belva Fahrozi},
  journal = {[Target venue — TBD]},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The TS2Vec encoder is used under its original license:
https://github.com/yuezhihan/ts2vec/blob/main/LICENSE
