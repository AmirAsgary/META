# META: Medoid-to-Ensemble Topological Autoencoder

**Dynamics-aware inverse protein folding via cochain complexes**

## Architecture

| Component | Description |
|-----------|-------------|
| Representation | 4-rank cochain complex: residues → edges → bends → torsions |
| Message passing | Hybrid sparse attention + GCNN, configurable per layer |
| Multi-chain | Implicit via covalent bond encoding (no explicit chain labels) |
| Masking | Feature MLM + topology dropout with reconstruction |
| AR decoding | 2-layer TransformerDecoder + pointer network with REINFORCE |
| Biochemistry | Bend (3-residue) + torsion (4-residue) physicochemical conditioning |
| Dynamics | ANM ensemble MSF + pairwise distance variance (optional) |

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: fast test (no dynamics, no ProDy needed)
python train.py --data_dir ./pdbs --output_dir ./out \
  --layer_types conv --d_model 32 --n_layers 2 --epochs 5

# Step 2: full training with dynamics
python train.py --data_dir ./pdbs --output_dir ./out \
  --compute_dynamics --n_layers 4 --layer_types attn,conv,conv,conv \
  --d_model 128 --n_heads 4 --use_ar --use_pointer --mixed_precision

# Step 3: large-scale with LMDB
python preprocess.py --pdb_dir ./pdbs --cache_dir ./cache --compute_dynamics --build_lmdb
python train.py --data_dir ./pdbs --output_dir ./out --lmdb_path ./cache/train.lmdb \
  --n_layers 8 --layer_types attn,conv,conv,conv,conv,conv,conv,conv \
  --d_model 256 --n_heads 8 --use_ar --use_pointer --mixed_precision
```

## Training Curriculum (4 phases)

| Phase | Epochs | Sequence Loss | Dynamics γ | AR | Chunk Size |
|-------|--------|---------------|------------|-------|------------|
| 1: Encoder warm-up | 1–20 | L_seq (parallel) | 0 | off | N (parallel) |
| 2: Dynamics ramp | 21–50 | L_seq (parallel) | 0→target | off | N (parallel) |
| 3: AR annealing | 51–100 | L_AR (teacher forcing) | target | on | N/4→1 |
| 4: Fine-tuning | 100+ | L_AR (scheduled sampling 20%) | target | on | 1 |

## Data Pipeline

```
PDB/CIF → parse_structure → build_cochain_complex → .npz cache → [optional: build_lmdb → .lmdb]
                                ↓ (optional)
                        compute_anm_dynamics → _dyn.npz
```

Dynamics is **optional** (`--compute_dynamics`). Without it, `has_dynamics=0` and `L_dyn=0` automatically. The network architecture stays intact.

## Tests

```bash
python test_v6.py  # 26 tests: layers, decoders, fixes, batch>1, global/per-protein AR
```

## File Structure

```
META/
├── src/
│   ├── __init__.py
│   ├── utils.py          # Geometry, encodings, physicochemistry
│   ├── model_utils.py    # Model, layers, decoders, loss
│   └── processing.py     # Parsing, cochain complex, LMDB, datasets
├── train.py              # Training pipeline
├── train_utils.py        # Curriculum, scheduler, metrics, checkpointing
├── preprocess.py         # Standalone preprocessing + LMDB build
├── test_v6.py            # Comprehensive test suite (26 tests)
├── requirements.txt
└── README.md
```

---

## `train.py` Flags

### Data

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data_dir` | str | *required* | Directory containing PDB/CIF structure files |
| `--cache_dir` | str | `./cache` | Directory for preprocessed `.npz` cache files |
| `--lmdb_path` | str | `None` | Path to pre-built `.lmdb` file. If set, skips PDB processing and loads data directly from LMDB (fastest I/O) |
| `--output_dir` | str | `./output` | Directory for checkpoints, logs, `config.json`, `splits.json` |
| `--split_file` | str | `None` | Path to existing `splits.json`. If set, reuses those train/val/test splits instead of creating new ones |
| `--val_frac` | float | `0.1` | Fraction of data for validation set |
| `--test_frac` | float | `0.1` | Fraction of data for test set |
| `--max_len` | int | `500` | Maximum protein length (residues). Longer proteins are excluded at dataset init |
| `--min_len` | int | `30` | Minimum protein length. Shorter proteins are excluded during processing |
| `--edge_cutoff` | float | `8.0` | Cβ distance cutoff (Å) for edge construction in the cochain complex |
| `--compute_dynamics` | flag | off | Run ProDy ANM to compute MSF and pairwise distance variance targets. **Slow** (~5s per protein). Skip for initial testing — the model works fine without it, dynamics loss just becomes zero |
| `--n_modes` | int | `20` | Number of ANM normal modes to use for ensemble generation |
| `--n_conformers` | int | `1000` | Number of conformers to sample from ANM ensemble |
| `--n_workers_process` | int | `4` | Number of parallel workers for PDB preprocessing |
| `--per_chain` | flag | on | Process each chain separately (each chain = one training example with partner chains as structural context) |
| `--no_per_chain` | flag | — | Process entire complex as one example |

### Architecture

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--d_model` | int | `32` | Hidden dimension for all ranks. Use 32 for testing, 128–256 for production |
| `--n_heads` | int | `1` | Number of attention heads (for attention layers). Must divide `d_model` |
| `--n_layers` | int | `1` | Number of META layers (each = intra-rank + up + down + FFN) |
| `--dropout` | float | `0.1` | Dropout probability throughout the model |
| `--layer_types` | str | `attn` | Per-layer type specification, comma-separated. Each entry is `attn` (sparse attention) or `conv` (graph convolution). Must have exactly `n_layers` entries or a single entry (broadcast). Examples: `conv` → all conv; `attn,conv,conv,conv` → 1 attention + 3 conv. Attention is ~4–6× slower but more expressive |
| `--use_ar` | flag | off | Enable autoregressive decoding. Activates the 2-layer TransformerDecoder head with causal masking. Without this, only parallel one-shot decoding (`seq_logits`) is used |
| `--use_pointer` | flag | off | Use learned pointer network for decoding order (requires `--use_ar`). Trained via REINFORCE with EMA baseline. Without this, random permutation is used |
| `--per_protein_ar` | flag | on | Run pointer network and MSF discretization independently per protein in the batch. **Required for correct behavior at batch_size > 1**. This is the default |
| `--global_ar` | flag | — | Run pointer/AR globally on concatenated batch. Faster (avoids per-protein loop) but **only correct at batch_size=1**. Cross-protein attention is still blocked by the causal mask, but the pointer produces a meaningless global ordering |
| `--n_msf_bins` | int | `32` | Number of bins for discretizing predicted MSF values for the AR decoder conditioning |

### Masking (Self-Supervised Regularizers)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mask_ratio` | float | `0.0` | Fraction of cochain features to mask per rank (MLM-style). `0.0` = off, `0.15` = typical. Masked positions are replaced with learned mask tokens; a reconstruction head predicts the original features. Active only during training |
| `--topo_mask_ratio` | float | `0.0` | Fraction of neighbourhood/incidence edges to drop. `0.0` = off, `0.1` = typical. A bilinear scorer reconstructs dropped connections via BCE. Provides topology-aware regularization |

### Loss Weights

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--alpha` | float | `1.0` | Weight for sequence prediction loss (L_seq or L_AR depending on curriculum phase) |
| `--beta` | float | `0.5` | Weight for biochemistry loss (bend + torsion physicochemical property prediction) |
| `--gamma` | float | `0.5` | **Target** weight for dynamics loss (MSF + pairwise distance variance). Actual weight is 0 in Phase 1, linearly ramped in Phase 2, and reaches this value in Phase 3+. If `--compute_dynamics` is not set, this has no effect (all proteins have `has_dynamics=0`) |
| `--delta` | float | `0.1` | Weight for feature reconstruction loss (requires `--mask_ratio > 0`) |
| `--zeta` | float | `0.1` | Weight for topology reconstruction loss (requires `--topo_mask_ratio > 0`) |
| `--label_smoothing` | float | `0.1` | Label smoothing for the sequence cross-entropy loss |

### Training

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--epochs` | int | `100` | Total training epochs |
| `--batch_size` | int | `1` | Batch size. Proteins are concatenated PyG-style (not padded). Use `--per_protein_ar` (default) for batch > 1 with pointer network |
| `--lr` | float | `3e-4` | Peak learning rate for AdamW |
| `--weight_decay` | float | `0.01` | AdamW weight decay |
| `--grad_clip` | float | `1.0` | Maximum gradient norm (clipped per step) |
| `--warmup_steps` | int | `1000` | Linear LR warm-up steps before cosine decay |
| `--num_workers` | int | `2` | DataLoader workers for data loading |
| `--pin_memory` | flag | on | Pin memory for faster GPU transfer |

### Curriculum Phases

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--phase1_epochs` | int | `20` | Phase 1 duration: encoder warm-up with parallel L_seq + L_biochem, γ=0 |
| `--phase2_epochs` | int | `30` | Phase 2 duration: linearly ramp γ from 0 to `--gamma` target |
| `--phase3_epochs` | int | `50` | Phase 3 duration: switch to L_AR with teacher forcing, activate pointer, anneal chunk size from N/4 → 1 |
| `--sched_sample_ratio` | float | `0.2` | Scheduled sampling ratio in Phase 4 (fraction of positions that use model predictions instead of ground truth during teacher forcing) |

Phase 4 starts automatically after Phase 3 ends (epoch > p1+p2+p3) and continues until `--epochs`.

### Misc

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mixed_precision` | flag | off | Use FP16 automatic mixed precision on CUDA. Halves memory for activations, ~1.5× speedup |
| `--resume` | str | `None` | Path to checkpoint `.pt` file to resume training from |
| `--save_every` | int | `5` | Save checkpoint every N epochs |
| `--patience` | int | `15` | Early stopping patience (epochs without validation loss improvement) |
| `--log_every` | int | `50` | Log training metrics every N steps |
| `--eval_every` | int | `1` | Run validation every N epochs |
| `--log_level` | str | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`) |
| `--device` | str | `auto` | Device: `auto` (CUDA if available, else CPU), `cuda`, `cpu`, `cuda:0`, etc. |
| `--seed` | int | `42` | Random seed for reproducibility |

---

## `preprocess.py` Flags

Standalone script for data preprocessing. Use when you want to preprocess once and train multiple times, or to build LMDB for faster I/O.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pdb_dir` | str | *required* | Directory with PDB/CIF files |
| `--cache_dir` | str | `./cache` | Output directory for `.npz` cache files |
| `--output_dir` | str | `./output` | Output directory for `splits.json` |
| `--compute_dynamics` | flag | off | Run ProDy ANM (creates `_dyn.npz` files alongside structure `.npz`) |
| `--build_lmdb` | flag | off | After processing, merge all `.npz` files into a single `.lmdb` file for fast streaming |
| `--lmdb_path` | str | `None` | Custom LMDB output path. Default: `{cache_dir}/train.lmdb` |
| `--n_modes` | int | `20` | ANM normal modes |
| `--n_conformers` | int | `1000` | ANM conformers to sample |
| `--edge_cutoff` | float | `8.0` | Cβ distance cutoff (Å) |
| `--n_workers` | int | `4` | Parallel processing workers |
| `--val_frac` | float | `0.1` | Validation fraction for splits |
| `--test_frac` | float | `0.1` | Test fraction for splits |
| `--per_chain` | flag | on | Process each chain separately |
| `--no_per_chain` | flag | — | Process as whole complex |
| `--min_len` | int | `30` | Minimum chain length |
| `--max_len` | int | `500` | Maximum chain length |
| `--seed` | int | `42` | Random seed for splits |
| `--log_level` | str | `INFO` | Logging level |

---

## Example Configurations

```bash
# Minimal test run (CPU, ~5 min on 100 PDBs)
python train.py --data_dir ./pdbs --output_dir ./test_run \
  --d_model 32 --n_layers 1 --layer_types conv \
  --epochs 5 --batch_size 1

# Medium run with AR (GPU recommended)
python train.py --data_dir ./pdbs --output_dir ./medium_run \
  --d_model 64 --n_heads 2 --n_layers 3 --layer_types attn,conv,conv \
  --use_ar --epochs 50 --batch_size 4 --mixed_precision

# Full production run with everything enabled
python train.py --data_dir ./pdbs --output_dir ./full_run \
  --lmdb_path ./cache/train.lmdb \
  --d_model 256 --n_heads 8 --n_layers 8 \
  --layer_types attn,conv,conv,conv,conv,conv,conv,conv \
  --use_ar --use_pointer \
  --compute_dynamics --mask_ratio 0.15 --topo_mask_ratio 0.1 \
  --delta 0.3 --zeta 0.2 \
  --epochs 150 --batch_size 4 --mixed_precision \
  --phase1_epochs 20 --phase2_epochs 30 --phase3_epochs 50

# Resume from checkpoint
python train.py --data_dir ./pdbs --output_dir ./full_run \
  --resume ./full_run/best_model.pt \
  --epochs 200  # continues from saved epoch
```
