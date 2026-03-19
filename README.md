# META: Medoid-to-Ensemble Topological Autoencoder

**Generative Inverse Folding for Protein Dynamics via Cochain Complexes**

## Architecture Summary

| Feature | Description |
|---------|-------------|
| Representation | 4-rank cochain complex (residues/edges/bends/torsions) |
| Message passing | Hybrid attention + GCNN, configurable per layer |
| Multi-chain | Implicit via covalent bond encoding (no explicit chain labels) |
| Masking | Feature MLM + topology dropout with reconstruction objectives |
| AR decoding | Pointer network for learned decoding order + chunked decoding |
| Biochemistry | Bend (3-residue) + torsion (4-residue) physicochemical conditioning |
| Dynamics | ANM ensemble MSF + pairwise distance variance prediction |

## Cochain Complex (4 Ranks)

| Rank | Name | Feature | Dim |
|------|------|---------|-----|
| 0 | Residues | dihedrals(6) + bond angle RBF(16) + SASA(1) | 23 |
| 1 | Edges | dist RBF(16) + local dir(3) + seq sep(16) + covalent(2) | 37 |
| 2 | Bends | cos(bend angle at j) | 1 |
| 3 | Torsions | sin/cos(dihedral) | 2 |

**No DSSP, no explicit chain labels** — the model infers both from continuous geometry and covalent bond topology.

## Key Design Choices

- **Implicit multi-chain**: Complexes are single higher-order graphs. Chain boundaries are encoded only through covalent bond indicators. Cross-chain bends/torsions emerge naturally at protein-protein interfaces.
- **Pointer network AR**: Instead of random permutation, a learned pointer network decides which residues to decode first (most constrained by context). Supports chunked decoding (L residues per step).
- **Torsion biochemistry**: 4-residue microenvironment properties (hydrophobicity, charge, volume, H-bond capacity) predicted from rank-3 latents and used as AR conditioning signal.

## Quick Start

```bash
pip install -r requirements.txt

# Minimal (conv layers, no AR)
python train.py --data_dir ./pdbs --output_dir ./out --layer_types conv

# Full with pointer network AR + masking
python train.py --data_dir ./pdbs --output_dir ./out \
  --n_layers 4 --layer_types attn,conv,conv,conv --d_model 128 --n_heads 4 \
  --use_ar --use_pointer --chunk_size 4 \
  --mask_ratio 0.15 --topo_mask_ratio 0.1 --delta 0.3 --zeta 0.2 \
  --compute_dynamics --mixed_precision
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--layer_types` | `attn` | Per-layer type: `attn` or `conv`, comma-separated |
| `--use_ar` | off | Enable autoregressive decoding |
| `--use_pointer` | off | Use pointer network for learned decoding order |
| `--chunk_size` | 1 | Positions decoded per pointer step (1=sequential) |
| `--mask_ratio` | 0.0 | Feature masking ratio |
| `--topo_mask_ratio` | 0.0 | Topology masking ratio |
| `--delta` | 0.1 | Feature reconstruction loss weight |
| `--zeta` | 0.1 | Topology reconstruction loss weight |

## Tests

```bash
python test_synthetic.py  # 12 tests: utils, parsing, cochain, model configs, masking, pointer net, torsion biochem
```
