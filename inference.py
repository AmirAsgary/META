"""inference.py — Fix #2: Complete inference pipeline per proposal §2.8.
6-step workflow: ANM ensemble → CC construction → encode → predict dynamics →
AR decode → rank candidates by log-probability.
Usage:
  python inference.py --checkpoint best_model.pt --pdb input.pdb --n_designs 100 --output designs/
  python inference.py --checkpoint best_model.pt --pdb input.pdb --chain A --temp 0.1 --top_p 0.95
"""
import argparse, json, logging, os, sys, time
import numpy as np
import torch
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
IDX_TO_AA = 'ACDEFGHIKLMNPQRSTVWY'
# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Ensemble generation (ANM → medoid)
# ══════════════════════════════════════════════════════════════════════════════
def generate_ensemble_and_medoid(pdb_path, chain_id=None, n_modes=20, n_conformers=1000,
                                  cutoff=15.0, edge_cutoff=8.0, min_len=30, max_len=500):
    """Run ANM on target backbone, extract medoid. Returns (backbone, dynamics) or (backbone, None)."""
    from src.processing import parse_structure, compute_anm_dynamics
    bb = parse_structure(pdb_path, chain_ids=chain_id, min_len=min_len)
    if bb is None: raise ValueError(f"Failed to parse {pdb_path} (chain={chain_id})")
    L = bb['n_res']
    if L < min_len or L > max_len: raise ValueError(f"Length {L} outside [{min_len},{max_len}]")
    logger.info(f"Parsed {pdb_path}: {L} residues, chain={chain_id or 'all'}")
    dyn = compute_anm_dynamics(bb, n_modes=n_modes, n_conformers=n_conformers, cutoff=cutoff)
    if dyn is not None:
        logger.info(f"ANM: {dyn['n_modes']} modes, medoid_idx={dyn['medoid_idx']}")
        mbb = {**bb, 'CA': dyn['medoid_coords']}
    else:
        logger.warning("ANM unavailable (ProDy not installed?), using original structure as medoid")
        mbb = bb
    return mbb, dyn
# ══════════════════════════════════════════════════════════════════════════════
# Step 2: CC construction from medoid
# ══════════════════════════════════════════════════════════════════════════════
def build_cc(backbone, dynamics, edge_cutoff=8.0):
    """Build cochain complex + compute dynamics targets if available."""
    from src.processing import build_cochain_complex, compute_pairwise_dist_var_vectorized
    cc = build_cochain_complex(backbone, edge_cutoff=edge_cutoff)
    if dynamics is not None:
        pv = compute_pairwise_dist_var_vectorized(backbone, dynamics, cc['edge_src'], cc['edge_dst'])
        cc['msf'] = dynamics['msf'].astype(np.float32)
        cc['msf_empirical'] = dynamics['msf_empirical'].astype(np.float32)
        cc['pair_var'] = pv.astype(np.float32)
        cc['has_dynamics'] = np.array([1], dtype=np.int64)
    else:
        cc['msf'] = np.zeros(cc['n_res'], dtype=np.float32)
        cc['pair_var'] = np.zeros(cc['n_edges'], dtype=np.float32)
        cc['has_dynamics'] = np.array([0], dtype=np.int64)
    logger.info(f"CC: {cc['n_res']} nodes, {cc['n_edges']} edges, "
                f"{cc['n_bends']} bends, {cc['n_torsions']} torsions")
    return cc
# ══════════════════════════════════════════════════════════════════════════════
# Step 3-4: Load model, encode, predict dynamics
# ══════════════════════════════════════════════════════════════════════════════
def load_model(ckpt_path, device='cpu'):
    """Load trained META model from checkpoint."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    # try to recover model config from checkpoint
    cfg = ck.get('config', ck.get('args', {}))
    if isinstance(cfg, argparse.Namespace): cfg = vars(cfg)
    from src.model_utils import METAModel
    model = METAModel(
        d_model=cfg.get('d_model', 256), n_heads=cfg.get('n_heads', 8),
        n_layers=cfg.get('n_layers', 8), dropout=0.0,  # no dropout at inference
        d_node=cfg.get('d_node', 23), d_edge=cfg.get('d_edge', 37),
        d_bend=cfg.get('d_bend', 1), d_torsion=cfg.get('d_torsion', 2),
        use_ar=True, n_msf_bins=cfg.get('n_msf_bins', 32),
        layer_types=cfg.get('layer_types', 'attn,conv,conv,conv,conv,conv,conv,conv'),
        mask_ratio=0.0, topo_mask_ratio=0.0,
        use_pointer=cfg.get('use_pointer', True),
        chunk_size=1, per_protein_ar=True
    ).to(device)
    model.load_state_dict(ck['model_state'], strict=False)
    model.eval()
    logger.info(f"Model loaded from {ckpt_path} (epoch={ck.get('epoch','?')})")
    return model, cfg
def cc_to_batch(cc, device='cpu'):
    """Convert numpy CC dict to a single-protein batch dict on device."""
    from src.data import collate_fn  # reuse existing collate if available
    b = {}
    for k, v in cc.items():
        if isinstance(v, np.ndarray):
            if v.dtype in (np.float32, np.float64): b[k] = torch.tensor(v, dtype=torch.float32, device=device)
            elif v.dtype in (np.int32, np.int64): b[k] = torch.tensor(v, dtype=torch.long, device=device)
            else: b[k] = torch.tensor(v, device=device)
        else: b[k] = v
    N = cc['n_res']
    # add batch vectors (single protein)
    b['node_batch'] = torch.zeros(N, dtype=torch.long, device=device)
    b['edge_batch'] = torch.zeros(cc['n_edges'], dtype=torch.long, device=device)
    b['bend_batch'] = torch.zeros(cc['n_bends'], dtype=torch.long, device=device)
    b['torsion_batch'] = torch.zeros(cc['n_torsions'], dtype=torch.long, device=device)
    b['has_dynamics'] = torch.tensor(cc['has_dynamics'], dtype=torch.long, device=device)
    # ensure required keys exist
    if 'msf' not in b: b['msf'] = torch.zeros(N, device=device)
    if 'pair_var' not in b: b['pair_var'] = torch.zeros(cc['n_edges'], device=device)
    return b
# ══════════════════════════════════════════════════════════════════════════════
# Step 5: AR decoding — generate S candidate sequences
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_sequences(model, batch, n_designs=100, temperature=0.1, top_p=0.9,
                       mask_ratio=0.0, device='cpu'):
    """Generate n_designs candidate sequences via AR decoding.
    Returns list of dicts: [{seq, log_prob, perm}, ...]"""
    model.eval()
    candidates = []
    h0_cache = None
    for s in range(n_designs):
        # optional inference-time masking for diversity (§2.5.3)
        if mask_ratio > 0: model.mask_ratio = mask_ratio
        else: model.mask_ratio = 0.0
        # encode
        pred = model(batch)
        h0 = pred.get('_h0')
        # get encoder outputs needed for AR
        msf_pred = pred['msf_pred']
        # pool bend/torsion context
        h_states = pred.get('_h_states')  # if model exposes intermediate states
        # use the AR decoder's generate method
        if hasattr(model, 'ar_decoder') and hasattr(model.ar_decoder, 'generate'):
            nb = batch.get('node_batch')
            msf_bins = model._discretize_msf(msf_pred.detach(), nb)
            bend_ctx = model._pool_context_from_pred(pred, 'bends', batch)
            torsion_ctx = model._pool_context_from_pred(pred, 'torsions', batch)
            # get encoding
            h0_enc = pred.get('_node_latent', pred['seq_logits'].new_zeros(1))
            # generate permutation via pointer or random
            N = batch['node_feat'].shape[0]
            if hasattr(model, 'pointer_net') and model.use_pointer:
                perm, _ = model.pointer_net(h0_enc, chunk_size=1)
            else:
                perm = torch.randperm(N, device=device)
            seq, log_probs = model.ar_decoder.generate(
                h0_enc, msf_bins, bend_ctx, torsion_ctx, perm, temp=temperature, top_p=top_p)
            total_log_prob = log_probs.sum().item()
            seq_str = ''.join(IDX_TO_AA[i] for i in seq.cpu().numpy() if i < 20)
        else:
            # fallback: use one-shot logits with sampling
            logits = pred['seq_logits'] / temperature
            probs = torch.softmax(logits, -1)
            seq = torch.multinomial(probs, 1).squeeze(-1)
            total_log_prob = torch.log(probs.gather(1, seq.unsqueeze(1)) + 1e-12).sum().item()
            seq_str = ''.join(IDX_TO_AA[i] for i in seq.cpu().numpy() if i < 20)
            perm = torch.arange(len(seq))
        candidates.append({'seq': seq_str, 'log_prob': round(total_log_prob, 4),
                           'perm': perm.cpu().numpy().tolist(), 'sample_idx': s})
        if (s + 1) % 10 == 0: logger.info(f"  Generated {s+1}/{n_designs} sequences")
    return candidates
# ══════════════════════════════════════════════════════════════════════════════
# Alternative: direct generation bypassing model.forward() for efficiency
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_sequences_efficient(model, batch, n_designs=100, temperature=0.1, top_p=0.9,
                                 mask_ratio=0.0, device='cpu'):
    """Efficient generation: encode once, decode S times with different seeds.
    This avoids re-running the encoder for every candidate."""
    model.eval()
    # single encoder pass
    model.mask_ratio = mask_ratio if mask_ratio > 0 else 0.0
    orig_ar = model.use_ar; model.use_ar = False  # skip AR in forward
    pred = model(batch)
    model.use_ar = orig_ar
    h0 = pred['seq_logits'].new_zeros(1)  # placeholder
    # extract encoder hidden states — need to call encode directly
    orig_feats = [batch['node_feat'], batch['edge_feat'], batch['bend_feat'], batch['torsion_feat']]
    topo = model._build_topo(batch)
    masked_feats, masks, masked_topo = model.masker(orig_feats, topo, model.mask_ratio, 0.0, False)
    h = model.encode(masked_feats, masked_topo)
    h0_enc = h[0]  # node latent: (N, d_model)
    N = h0_enc.shape[0]
    # precompute conditioning
    bend_ctx = model._pool_context(h[2], 'bends', h[0], batch)
    torsion_ctx = model._pool_context(h[3], 'torsions', h[0], batch)
    nb = batch.get('node_batch')
    msf_pred = model.msf_decoder(h0_enc)
    msf_bins = model._discretize_msf(msf_pred.detach(), nb)
    candidates = []
    for s in range(n_designs):
        torch.manual_seed(s)  # reproducible diversity from different seeds
        if hasattr(model, 'pointer_net') and model.use_pointer:
            perm, _ = model.pointer_net(h0_enc, chunk_size=1)
        else:
            perm = torch.randperm(N, device=device)
        seq, log_probs = model.ar_decoder.generate(
            h0_enc, h[1], batch['edge_src'], batch['edge_dst'],
            msf_bins, bend_ctx, torsion_ctx, perm, temp=temperature, top_p=top_p)
        total_lp = log_probs.sum().item()
        seq_str = ''.join(IDX_TO_AA[i] for i in seq.cpu().numpy() if i < 20)
        candidates.append({'seq': seq_str, 'log_prob': round(total_lp, 4),
                           'perm': perm.cpu().numpy().tolist(), 'sample_idx': s})
        if (s + 1) % 25 == 0: logger.info(f"  Generated {s+1}/{n_designs} sequences")
    return candidates
# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Rank candidates
# ══════════════════════════════════════════════════════════════════════════════
def rank_candidates(candidates):
    """Rank by descending joint log-probability."""
    return sorted(candidates, key=lambda c: c['log_prob'], reverse=True)
# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════
def write_fasta(candidates, output_path, pdb_name='design'):
    """Write ranked candidates as FASTA."""
    with open(output_path, 'w') as f:
        for i, c in enumerate(candidates):
            f.write(f">{pdb_name}_design_{i:04d} log_prob={c['log_prob']:.4f} sample={c['sample_idx']}\n")
            f.write(c['seq'] + '\n')
    logger.info(f"Wrote {len(candidates)} sequences to {output_path}")
def write_json(candidates, output_path, metadata=None):
    """Write full results as JSON."""
    out = {'metadata': metadata or {}, 'n_designs': len(candidates), 'candidates': candidates}
    with open(output_path, 'w') as f: json.dump(out, f, indent=2)
    logger.info(f"Wrote results to {output_path}")
# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='META inference: dynamics-aware sequence design')
    # input
    p.add_argument('--pdb', required=True, help='Input PDB/CIF file (target backbone)')
    p.add_argument('--chain', default=None, help='Chain ID(s) to process (default: all)')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint (.pt)')
    # ANM
    p.add_argument('--n_modes', type=int, default=20, help='Number of ANM normal modes')
    p.add_argument('--n_conformers', type=int, default=1000, help='Number of ANM conformers')
    p.add_argument('--anm_cutoff', type=float, default=15.0, help='ANM distance cutoff')
    p.add_argument('--edge_cutoff', type=float, default=8.0, help='CB edge cutoff')
    # generation
    p.add_argument('--n_designs', type=int, default=100, help='Number of candidate sequences')
    p.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling threshold')
    p.add_argument('--mask_ratio', type=float, default=0.0, help='Inference-time masking for diversity (0=off, 0.05-0.15=typical)')
    # output
    p.add_argument('--output', default='./designs', help='Output directory')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    pdb_name = Path(args.pdb).stem
    t_start = time.time()
    # ── Step 1: Ensemble generation ──
    logger.info("Step 1: Generating ANM ensemble and extracting medoid...")
    t0 = time.time()
    backbone, dynamics = generate_ensemble_and_medoid(
        args.pdb, chain_id=args.chain, n_modes=args.n_modes,
        n_conformers=args.n_conformers, cutoff=args.anm_cutoff, edge_cutoff=args.edge_cutoff)
    logger.info(f"  Ensemble generation: {time.time()-t0:.2f}s")
    # ── Step 2: CC construction ──
    logger.info("Step 2: Building cochain complex...")
    t0 = time.time()
    cc = build_cc(backbone, dynamics, edge_cutoff=args.edge_cutoff)
    logger.info(f"  CC construction: {time.time()-t0:.2f}s")
    # ── Step 3: Load model + encode ──
    logger.info("Step 3: Loading model and encoding...")
    t0 = time.time()
    model, cfg = load_model(args.checkpoint, device)
    batch = cc_to_batch(cc, device)
    logger.info(f"  Model loading: {time.time()-t0:.2f}s")
    # ── Step 4: Dynamicity prediction (happens inside generate) ──
    logger.info("Step 4-5: Predicting dynamics and generating sequences...")
    t0 = time.time()
    candidates = generate_sequences_efficient(
        model, batch, n_designs=args.n_designs, temperature=args.temperature,
        top_p=args.top_p, mask_ratio=args.mask_ratio, device=device)
    logger.info(f"  Generation ({args.n_designs} seqs): {time.time()-t0:.2f}s")
    # ── Step 6: Rank ──
    logger.info("Step 6: Ranking candidates...")
    ranked = rank_candidates(candidates)
    # report top-5
    logger.info("Top 5 designs:")
    for i, c in enumerate(ranked[:5]):
        logger.info(f"  #{i+1}: log_prob={c['log_prob']:.4f} seq={c['seq'][:50]}...")
    # ── Write output ──
    metadata = {'pdb': args.pdb, 'chain': args.chain, 'n_designs': args.n_designs,
                'temperature': args.temperature, 'top_p': args.top_p,
                'mask_ratio': args.mask_ratio, 'n_residues': cc['n_res'],
                'has_dynamics': bool(dynamics is not None),
                'total_time_s': round(time.time() - t_start, 2),
                'checkpoint': args.checkpoint}
    write_fasta(ranked, os.path.join(args.output, f'{pdb_name}_designs.fasta'), pdb_name)
    write_json(ranked, os.path.join(args.output, f'{pdb_name}_designs.json'), metadata)
    # write MSF prediction for analysis
    if dynamics is not None:
        model.eval()
        with torch.no_grad():
            pred = model(batch)
        msf_pred = pred['msf_pred'].cpu().numpy()
        np.savez(os.path.join(args.output, f'{pdb_name}_dynamics.npz'),
                 msf_pred=msf_pred, msf_gt=cc.get('msf', np.zeros_like(msf_pred)))
        logger.info(f"  MSF predictions saved")
    logger.info(f"Done. Total time: {time.time()-t_start:.2f}s. Output: {args.output}")
if __name__ == '__main__': main()
