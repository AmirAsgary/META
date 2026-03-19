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

Dynamics is **optional** (`--compute_dynamics` flag). Without it, has_dynamics=0 and L_dyn=0 automatically. The full network architecture stays intact — MSF/PairVar decoders still exist, they just predict dummy zeros that aren't penalized.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--layer_types` | `attn` | Per-layer type: `attn` or `conv`, comma-separated |
| `--use_ar` | off | Enable autoregressive decoding |
| `--use_pointer` | off | Pointer network for learned decoding order |
| `--compute_dynamics` | off | Run ProDy ANM (slow, skip for quick tests) |
| `--lmdb_path` | None | Pre-built LMDB for fast streaming |
| `--mask_ratio` | 0.0 | Feature masking ratio (0.15 typical) |
| `--topo_mask_ratio` | 0.0 | Topology masking ratio (0.1 typical) |
| `--mixed_precision` | off | FP16 training on CUDA |

## Tests

```bash
python test_v6.py  # 23 tests covering all layers, decoders, fixes, and data flow
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
├── test_v6.py            # Comprehensive test suite
├── requirements.txt
└── README.md
```
