"""META v6.2 comprehensive test suite.
Tests all layers + SparseARDecoder + scheduled sampling + generate + batch>1.
Run: python test_v6.py"""
import sys, os, time, traceback, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PASS = "\033[92mPASS\033[0m"; FAIL = "\033[91mFAIL\033[0m"; WARN = "\033[93mWARN\033[0m"
results = []; t0_global = time.time()
def report(name, passed, detail="", elapsed=0.0):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  [{status}] {name} ({elapsed:.3f}s) {detail}")
def make_batch(n_res=30, n_edges=100, n_bends=80, n_torsions=30, d_node=23, d_edge=37, dev='cpu'):
    import torch; b = {}
    b['node_feat'] = torch.randn(n_res, d_node, device=dev)
    b['edge_feat'] = torch.randn(n_edges, d_edge, device=dev)
    b['bend_feat'] = torch.randn(n_bends, 1, device=dev)
    b['torsion_feat'] = torch.randn(n_torsions, 2, device=dev)
    b['seq_idx'] = torch.randint(0, 20, (n_res,), device=dev)
    b['biochem_targets'] = torch.randn(n_bends, 4, device=dev)
    b['torsion_biochem_targets'] = torch.randn(n_torsions, 4, device=dev)
    b['msf'] = torch.rand(n_res, device=dev)*10; b['pair_var'] = torch.rand(n_edges, device=dev)
    b['has_dynamics'] = torch.tensor([1], device=dev)
    s = torch.randint(0, n_res, (n_edges,), device=dev); d = torch.randint(0, n_res, (n_edges,), device=dev)
    b['edge_src'] = s; b['edge_dst'] = d; b['nbr0_src'] = s; b['nbr0_dst'] = d
    b['nbr1_src'] = torch.randint(0, n_edges, (min(n_edges*3,800),), device=dev)
    b['nbr1_dst'] = torch.randint(0, n_edges, (min(n_edges*3,800),), device=dev)
    b['nbr2_src'] = torch.randint(0, max(n_bends,1), (min(n_bends*2,400),), device=dev)
    b['nbr2_dst'] = torch.randint(0, max(n_bends,1), (min(n_bends*2,400),), device=dev)
    b['nbr3_src'] = torch.randint(0, max(n_torsions,1), (min(n_torsions*2,200),), device=dev)
    b['nbr3_dst'] = torch.randint(0, max(n_torsions,1), (min(n_torsions*2,200),), device=dev)
    b['inc_01_edge'] = torch.cat([torch.arange(n_edges, device=dev)]*2)
    b['inc_01_node'] = torch.cat([s, d])
    b['inc_12_bend'] = torch.arange(n_bends, device=dev).repeat(2)
    b['inc_12_edge'] = torch.randint(0, n_edges, (n_bends*2,), device=dev)
    b['inc_23_torsion'] = torch.arange(n_torsions, device=dev).repeat(2)
    b['inc_23_bend'] = torch.randint(0, max(n_bends,1), (n_torsions*2,), device=dev)
    b['bends'] = torch.randint(0, n_res, (n_bends, 3), device=dev)
    b['torsions'] = torch.randint(0, n_res, (n_torsions, 4), device=dev)
    b['n_res'] = torch.tensor([n_res]); b['n_edges'] = torch.tensor([n_edges])
    b['n_bends'] = torch.tensor([n_bends]); b['n_torsions'] = torch.tensor([n_torsions])
    b['node_batch'] = torch.zeros(n_res, dtype=torch.long, device=dev)
    b['edge_batch'] = torch.zeros(n_edges, dtype=torch.long, device=dev)
    b['bend_batch'] = torch.zeros(n_bends, dtype=torch.long, device=dev)
    b['torsion_batch'] = torch.zeros(n_torsions, dtype=torch.long, device=dev)
    return b
# ══════════════════════════════════════════════════════════════════════════════
# Gradient checker — aware of intentionally unused parameter groups:
#   skip_mask_tokens:  masker.mask_tokens unused when mask_ratio=0
#   skip_seq_decoder:      seq_decoder unused when AR active (Fix #6 zeros it)
#   skip_unused_heads:     recon/topo heads unused when masking=0
#   skip_dynamics_decoders: msf_decoder+pair_var_decoder unused when gamma=0
#                           (msf_pred.detach() in _discretize_msf kills the other path)
# ══════════════════════════════════════════════════════════════════════════════
def _check_grads(model, skip_unused_heads=True, skip_seq_decoder=False,
                 skip_mask_tokens=False, skip_dynamics_decoders=False):
    no_grad = []
    skip_prefixes = []
    if skip_unused_heads: skip_prefixes += ['recon_heads', 'topo_nbr_heads', 'topo_inc_heads']
    if skip_seq_decoder: skip_prefixes += ['seq_decoder']
    if skip_mask_tokens: skip_prefixes += ['masker.mask_tokens']
    if skip_dynamics_decoders: skip_prefixes += ['msf_decoder', 'pair_var_decoder']
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            if any(name.startswith(s) for s in skip_prefixes): continue
            no_grad.append(name)
    return len(no_grad) == 0, no_grad
# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Scatter utilities
# ══════════════════════════════════════════════════════════════════════════════
def test_scatter_utils():
    t0 = time.time(); import torch
    from src.model_utils import scatter_softmax_2d, scatter_add_3d, scatter_mean_2d
    N, H = 10, 4; E = 30
    src = torch.randn(E, H); idx = torch.randint(0, N, (E,))
    sm = scatter_softmax_2d(src, idx, N)
    sums = torch.zeros(N, H).scatter_add_(0, idx.unsqueeze(1).expand_as(sm), sm)
    has_edge = torch.zeros(N, H).scatter_add_(0, idx.unsqueeze(1).expand_as(src), torch.ones_like(src)) > 0
    assert torch.allclose(sums[has_edge], torch.ones_like(sums[has_edge]), atol=1e-5), "softmax sums != 1"
    simple_src = torch.ones(6, 3); simple_idx = torch.tensor([0,0,0,1,1,2])
    mean = scatter_mean_2d(simple_src, simple_idx, 3)
    assert torch.allclose(mean, torch.ones(3, 3), atol=1e-5), "scatter_mean incorrect"
    report("scatter_utils", True, "softmax sums OK, scatter_mean OK", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: SparseNeighbourhoodSelfAttn
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_self_attn():
    t0 = time.time(); import torch
    from src.model_utils import SparseNeighbourhoodSelfAttn
    d, N, E, H = 64, 50, 200, 4
    X = torch.randn(N, d, requires_grad=True)
    src = torch.randint(0, N, (E,)); dst = torch.randint(0, N, (E,))
    attn = SparseNeighbourhoodSelfAttn(d, H, 0.0)
    out = attn(X, src, dst, N)
    ok_shape = out.shape == (N, d)
    out.sum().backward(); ok_grad = X.grad is not None and X.grad.shape == (N, d)
    out_empty = attn(X, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), N)
    ok_empty = torch.all(out_empty == 0)
    report("SparseNeighbourhoodSelfAttn", ok_shape and ok_grad and ok_empty.item(),
           f"shape={out.shape}, grad={ok_grad}, empty={ok_empty.item()}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: SparseGraphConv
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_graph_conv():
    t0 = time.time(); import torch
    from src.model_utils import SparseGraphConv
    d, N, E = 64, 50, 200
    X = torch.randn(N, d, requires_grad=True)
    src = torch.randint(0, N, (E,)); dst = torch.randint(0, N, (E,))
    conv = SparseGraphConv(d, 0.0)
    out = conv(X, src, dst, N)
    ok_shape = out.shape == (N, d)
    out.sum().backward(); ok_grad = X.grad is not None
    report("SparseGraphConv", ok_shape and ok_grad, f"shape={out.shape}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Cross-attention and incidence conv
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_ops():
    t0 = time.time(); import torch
    from src.model_utils import SparseTopologicalCrossAttn, SparseIncidenceConv
    d, Nt, Ns, E, H = 32, 40, 60, 100, 2
    X_tgt = torch.randn(Nt, d, requires_grad=True); X_src = torch.randn(Ns, d, requires_grad=True)
    inc_tgt = torch.randint(0, Nt, (E,)); inc_src = torch.randint(0, Ns, (E,))
    ca = SparseTopologicalCrossAttn(d, H, 0.0)
    out_ca = ca(X_tgt, X_src, inc_tgt, inc_src, Nt)
    ok_ca = out_ca.shape == (Nt, d)
    out_ca.sum().backward(); ok_grad_ca = X_tgt.grad is not None and X_src.grad is not None
    ic = SparseIncidenceConv(d, 0.0)
    X_tgt2 = torch.randn(Nt, d, requires_grad=True); X_src2 = torch.randn(Ns, d, requires_grad=True)
    out_ic = ic(X_tgt2, X_src2, inc_tgt, inc_src, Nt)
    ok_ic = out_ic.shape == (Nt, d)
    out_ic.sum().backward(); ok_grad_ic = X_tgt2.grad is not None
    report("CrossAttn+IncConv", ok_ca and ok_ic and ok_grad_ca and ok_grad_ic,
           f"attn={out_ca.shape}, conv={out_ic.shape}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: METALayer full forward (4 ranks, up/down pass)
# ══════════════════════════════════════════════════════════════════════════════
def test_meta_layer():
    t0 = time.time(); import torch
    from src.model_utils import METALayer
    d = 32; layer_attn = METALayer(d, 2, 0.0, use_attention=True)
    layer_conv = METALayer(d, 2, 0.0, use_attention=False)
    batch = make_batch(20, 60, 40, 15)
    topo = {k: batch[k] for k in ['nbr0_src','nbr0_dst','nbr1_src','nbr1_dst','nbr2_src','nbr2_dst','nbr3_src','nbr3_dst',
                                   'inc_01_edge','inc_01_node','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']}
    h = [torch.randn(20, d), torch.randn(60, d), torch.randn(40, d), torch.randn(15, d)]
    for name, layer in [("attn", layer_attn), ("conv", layer_conv)]:
        h_in = [x.clone().requires_grad_(True) for x in h]
        h_out = layer(h_in, topo)
        ok = all(h_out[r].shape == h_in[r].shape for r in range(4))
        h_out[0].sum().backward()
        if not ok: report(f"METALayer({name})", False, "shape mismatch"); return
    report("METALayer(attn+conv)", True, "4-rank up/down pass OK", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: SparseARDecoder — shape, gradient, 2 layers, causal mask
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_ar_decoder():
    t0 = time.time(); import torch
    from src.model_utils import SparseARDecoder
    dm, N, E = 32, 20, 60
    ar = SparseARDecoder(dm, n_aa=20, n_msf_bins=32, n_layers=2, dropout=0.0, scale=30)
    h_V_enc = torch.randn(N, dm, requires_grad=True)
    h_E_enc = torch.randn(E, dm)
    edge_src = torch.randint(0, N, (E,)); edge_dst = torch.randint(0, N, (E,))
    msf_bins = torch.randint(0, 32, (N,))
    bend_ctx = torch.randn(N, dm); torsion_ctx = torch.randn(N, dm)
    seq_gt = torch.randint(0, 20, (N,)); perm = torch.randperm(N)
    logits = ar(h_V_enc, h_E_enc, edge_src, edge_dst, msf_bins, bend_ctx, torsion_ctx, seq_gt, perm)
    ok_shape = logits.shape == (N, 20)
    logits.sum().backward(); ok_grad = h_V_enc.grad is not None
    ok_layers = len(ar.decoder_layers) == 2
    ok_has_head = hasattr(ar, 'head') and ar.head.out_features == 20
    ar.eval()
    with torch.no_grad():
        logits1 = ar(h_V_enc, h_E_enc, edge_src, edge_dst, msf_bins, bend_ctx, torsion_ctx, seq_gt, perm)
        seq_gt2 = seq_gt.clone(); seq_gt2[perm[-1]] = (seq_gt[perm[-1]] + 7) % 20
        logits2 = ar(h_V_enc, h_E_enc, edge_src, edge_dst, msf_bins, bend_ctx, torsion_ctx, seq_gt2, perm)
        first_pos = perm[0].item()
        ok_causal = torch.allclose(logits1[first_pos], logits2[first_pos], atol=1e-5)
    ar.train()
    passed = ok_shape and ok_grad and ok_layers and ok_has_head and ok_causal
    report("SparseARDecoder (shape+grad+causal)", passed,
           f"shape={ok_shape}, grad={ok_grad}, 2-layer={ok_layers}, causal={ok_causal}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 6b: _build_causal_mask correctness with hand-crafted example
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_causal_mask():
    t0 = time.time(); import torch
    from src.model_utils import SparseARDecoder
    ar = SparseARDecoder(16, n_layers=1)
    perm = torch.tensor([2, 0, 1])  # decode order: pos2(step0), pos0(step1), pos1(step2)
    edge_src = torch.tensor([0, 1, 0, 2]); edge_dst = torch.tensor([1, 0, 2, 0])
    mask = ar._build_causal_mask(perm, edge_src, edge_dst)
    # order[0]=1, order[1]=2, order[2]=0
    # edge 0->1: 1<2 True =>1, edge 1->0: 2<1 False =>0, edge 0->2: 1<0 False =>0, edge 2->0: 0<1 True =>1
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0]).unsqueeze(-1)
    ok_mask = torch.allclose(mask, expected)
    node_batch = torch.tensor([0, 0, 1])
    mask_batched = ar._build_causal_mask(perm, edge_src, edge_dst, node_batch)
    ok_cross = mask_batched[2].item() == 0.0 and mask_batched[3].item() == 0.0
    ok_same = mask_batched[0].item() == 1.0
    passed = ok_mask and ok_cross and ok_same
    report("SparseARDecoder causal mask", passed,
           f"mask_correct={ok_mask}, cross_blocked={ok_cross}, same_ok={ok_same}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 6c: SparseARDecoder scheduled sampling (two-pass)
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_sched_sampling():
    t0 = time.time(); import torch
    from src.model_utils import SparseARDecoder
    dm, N, E = 32, 15, 40
    ar = SparseARDecoder(dm, n_aa=20, n_msf_bins=32, n_layers=2, dropout=0.0)
    h_V = torch.randn(N, dm, requires_grad=True); h_E = torch.randn(E, dm)
    esrc = torch.randint(0, N, (E,)); edst = torch.randint(0, N, (E,))
    msf = torch.randint(0, 32, (N,)); bc = torch.randn(N, dm); tc = torch.randn(N, dm)
    seq = torch.randint(0, 20, (N,)); perm = torch.randperm(N)
    ar.train()
    torch.manual_seed(0)
    logits_no_ss = ar(h_V, h_E, esrc, edst, msf, bc, tc, seq, perm, sched_sample_ratio=0.0)
    torch.manual_seed(0)
    logits_ss = ar(h_V, h_E, esrc, edst, msf, bc, tc, seq, perm, sched_sample_ratio=1.0)
    ok_differs = not torch.allclose(logits_no_ss, logits_ss, atol=1e-4)
    ok_shape = logits_ss.shape == (N, 20)
    logits_ss.sum().backward(); ok_grad = h_V.grad is not None
    h_V2 = torch.randn(N, dm, requires_grad=True)
    torch.manual_seed(7)
    l1 = ar(h_V2, h_E, esrc, edst, msf, bc, tc, seq, perm, sched_sample_ratio=0.0)
    torch.manual_seed(7)
    l2 = ar(h_V2, h_E, esrc, edst, msf, bc, tc, seq, perm, sched_sample_ratio=0.0)
    ok_zero = torch.allclose(l1, l2, atol=1e-6)
    passed = ok_differs and ok_shape and ok_grad and ok_zero
    report("SparseARDecoder sched sampling", passed,
           f"differs={ok_differs}, shape={ok_shape}, grad={ok_grad}, zero_ratio_ok={ok_zero}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 6d: SparseARDecoder.generate()
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_ar_generate():
    t0 = time.time(); import torch
    from src.model_utils import SparseARDecoder
    dm, N, E = 32, 12, 30
    ar = SparseARDecoder(dm, n_aa=20, n_msf_bins=32, n_layers=2, dropout=0.0)
    ar.eval()
    h_V = torch.randn(N, dm); h_E = torch.randn(E, dm)
    esrc = torch.randint(0, N, (E,)); edst = torch.randint(0, N, (E,))
    msf = torch.randint(0, 32, (N,)); bc = torch.randn(N, dm); tc = torch.randn(N, dm)
    perm = torch.randperm(N)
    S, lp = ar.generate(h_V, h_E, esrc, edst, msf, bc, tc, perm, temp=0.1, top_p=0.9)
    ok_shape = S.shape == (N,) and lp.shape == (N,)
    ok_range = (S >= 0).all() and (S < 20).all()
    ok_all_decoded = (S != ar.n_aa).all()
    ok_logprobs = (lp <= 0).all()
    torch.manual_seed(0); S1, _ = ar.generate(h_V, h_E, esrc, edst, msf, bc, tc, perm, temp=1.0)
    torch.manual_seed(1); S2, _ = ar.generate(h_V, h_E, esrc, edst, msf, bc, tc, perm, temp=1.0)
    ok_diverse = not torch.equal(S1, S2)
    passed = ok_shape and ok_range.item() and ok_all_decoded.item() and ok_logprobs.item() and ok_diverse
    report("SparseARDecoder.generate()", passed,
           f"shape={ok_shape}, range={ok_range.item()}, decoded={ok_all_decoded.item()}, diverse={ok_diverse}",
           time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: REINFORCE loss for pointer network
# ══════════════════════════════════════════════════════════════════════════════
def test_reinforce_loss():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=2, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', use_ar=True, use_pointer=True, chunk_size=1)
    batch = make_batch(15, 40, 25, 10)
    model.train(); pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.0)
    total, ld = loss_fn(pred, batch, use_ar=True)
    total.backward()
    ptr_grads = [p.grad for p in model.pointer_net.parameters() if p.requires_grad and p.grad is not None]
    ok_ptr_grad = len(ptr_grads) > 0
    ok_ptr_loss = 'ptr_loss' in ld
    ok_nonzero_loss = abs(ld.get('ptr_loss', 0.0)) > 1e-12 if ok_ptr_loss else False
    ok_nonzero_grad = any(g.abs().sum() > 1e-10 for g in ptr_grads) if ok_ptr_grad else False
    passed = ok_ptr_grad and ok_ptr_loss and ok_nonzero_loss and ok_nonzero_grad
    report("REINFORCE pointer", passed,
           f"ptr_grads={len(ptr_grads)}, ptr_loss={ld.get('ptr_loss',0):.6f}, nonzero_grad={ok_nonzero_grad}", time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Chunk size annealing
# ══════════════════════════════════════════════════════════════════════════════
def test_chunk_annealing():
    t0 = time.time()
    from train_utils import TrainingCurriculum
    cur = TrainingCurriculum(p1=20, p2=30, p3=50, gamma_target=0.5)
    c_p1 = cur.get_chunk_size(5, max_len=200); ok_p1 = c_p1 == 200
    c_p2 = cur.get_chunk_size(30, max_len=200); ok_p2 = c_p2 == 200
    c_p3_start = cur.get_chunk_size(50, max_len=200); ok_p3_start = c_p3_start >= 40
    c_p3_end = cur.get_chunk_size(99, max_len=200); ok_p3_end = c_p3_end <= 3
    c_p4 = cur.get_chunk_size(110, max_len=200); ok_p4 = c_p4 == 1
    ok_gamma_0 = cur.get_gamma(5) == 0.0
    ok_gamma_mid = 0.0 < cur.get_gamma(35) < 0.5
    ok_gamma_full = cur.get_gamma(60) == 0.5
    passed = ok_p1 and ok_p2 and ok_p3_start and ok_p3_end and ok_p4 and ok_gamma_0 and ok_gamma_mid and ok_gamma_full
    report("Chunk annealing", passed,
           f"P1={c_p1}, P3_start={c_p3_start}, P3_end={c_p3_end}, P4={c_p4}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Scheduled sampling ratio from curriculum
# ══════════════════════════════════════════════════════════════════════════════
def test_scheduled_sampling():
    t0 = time.time()
    from train_utils import TrainingCurriculum
    cur = TrainingCurriculum(p1=20, p2=30, p3=50, sched_sample_ratio=0.2)
    ok_p1 = cur.get_sched_sample_ratio(5) == 0.0
    ok_p3 = cur.get_sched_sample_ratio(80) == 0.0
    ok_p4 = cur.get_sched_sample_ratio(110) == 0.2
    passed = ok_p1 and ok_p3 and ok_p4
    report("Scheduled sampling ratio", passed, f"P1={ok_p1}, P3={ok_p3}, P4={ok_p4}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: Clean negative sampling
# ══════════════════════════════════════════════════════════════════════════════
def test_clean_neg_sampling():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', topo_mask_ratio=0.3)
    batch = make_batch(20, 60, 40, 15)
    model.train(); pred = model(batch)
    ok_pairs = True
    for r in range(4):
        lo, la = pred['topo_nbr_logits'][r], pred['topo_nbr_labels'][r]
        if lo.shape != la.shape: ok_pairs = False
        if lo.shape[0] > 0:
            n_pos = (la > 0.5).sum().item(); n_neg = (la < 0.5).sum().item()
            if n_pos == 0 or n_neg == 0: ok_pairs = False
    loss_fn = METALoss(1.0, 0.5, 0.0, delta=0.0, zeta=0.5)
    total, ld = loss_fn(pred, batch)
    ok_loss = ld['topo_loss'] > 0; total.backward()
    report("Clean neg sampling", ok_pairs and ok_loss,
           f"pairs_ok={ok_pairs}, topo_loss={ld['topo_loss']:.4f}", time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Vectorized _pool_context
# ══════════════════════════════════════════════════════════════════════════════
def test_vectorized_pool():
    t0 = time.time(); import torch
    from src.model_utils import METAModel
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='conv')
    N0, M, d = 20, 40, 32
    h_rank = torch.randn(M, d); h0 = torch.randn(N0, d)
    bends = torch.tensor([[0,1,2],[0,1,3],[4,5,6]], dtype=torch.long)
    ctx = model._pool_context(h_rank[:3], 'bends', h0, {'bends': bends})
    ok_shape = ctx.shape == (N0, d)
    ok_nonzero = ctx[0].abs().sum() > 0; ok_zero = ctx[10].abs().sum() < 1e-6
    report("Vectorized _pool_context", ok_shape and ok_nonzero and ok_zero,
           f"shape={ctx.shape}, nonzero={ok_nonzero}, zero={ok_zero}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 16: Full model (conv, no AR, mask_ratio=0)
# skip_mask_tokens because mask_ratio=0 → tokens never enter forward
# ══════════════════════════════════════════════════════════════════════════════
def test_full_model_conv():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    from train_utils import compute_all_metrics
    model = METAModel(d_model=32, n_heads=1, n_layers=2, d_node=23, d_edge=37, layer_types='conv,conv')
    batch = make_batch(25, 70, 50, 20)
    pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.5); total, ld = loss_fn(pred, batch); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_mask_tokens=True)
    met = compute_all_metrics(pred, batch)
    report("Full model (conv,conv)", grads_ok,
           f"loss={ld['total']:.4f}, rec={met['recovery']:.3f}" + (f", missing={no_grad[:3]}" if not grads_ok else ""), time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 17: Full model (hybrid, no AR, mask_ratio=0)
# ══════════════════════════════════════════════════════════════════════════════
def test_full_model_hybrid():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=2, n_layers=3, d_node=23, d_edge=37, layer_types='attn,conv,conv')
    batch = make_batch(20, 60, 40, 15)
    pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.5); total, ld = loss_fn(pred, batch); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_mask_tokens=True)
    report("Full model (attn,conv,conv)", grads_ok,
           f"loss={ld['total']:.4f}" + (f", missing={no_grad[:3]}" if not grads_ok else ""), time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 18: Full model AR + pointer + all masking
# skip_seq_decoder: Fix #6 zeros seq_logits when AR active
# NOT skipping mask_tokens: mask_ratio=0.15 so they DO get grads
# ══════════════════════════════════════════════════════════════════════════════
def test_full_model_ar_pointer():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    from train_utils import compute_all_metrics
    model = METAModel(d_model=32, n_heads=2, n_layers=2, d_node=23, d_edge=37,
                      layer_types='attn,conv', use_ar=True, use_pointer=True, chunk_size=4,
                      mask_ratio=0.15, topo_mask_ratio=0.1)
    batch = make_batch(20, 60, 40, 15)
    model.train(); pred = model(batch)
    ok_keys = all(k in pred for k in ['seq_logits','ar_logits','biochem_pred','torsion_biochem_pred',
                                       'msf_pred','pair_var_pred','ptr_log_probs','perm',
                                       'recon_preds','recon_targets','recon_masks',
                                       'topo_nbr_logits','topo_inc_logits'])
    loss_fn = METALoss(1.0, 0.5, 0.5, delta=0.3, zeta=0.2)
    total, ld = loss_fn(pred, batch, use_ar=True); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_unused_heads=False, skip_seq_decoder=True)
    met = compute_all_metrics(pred, batch, use_ar=True)
    report("Full AR+Pointer+Masking", ok_keys and grads_ok,
           f"loss={ld['total']:.4f}, rec={met['recovery']:.3f}, ptr_loss={ld.get('ptr_loss',0):.4f}" +
           (f", missing={no_grad[:3]}" if not grads_ok else ""), time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 19: Full model with sched_sample_ratio > 0 (Phase 4 simulation)
# skip_seq_decoder: AR active | skip_mask_tokens: mask_ratio=0 | skip_dynamics: gamma=0
# ══════════════════════════════════════════════════════════════════════════════
def test_full_model_sched_sampling():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=2, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', use_ar=True, use_pointer=False)
    batch = make_batch(15, 40, 25, 10)
    model.train(); pred = model(batch, sched_sample_ratio=0.2)
    ok_has_ar = 'ar_logits' in pred
    ok_shape = pred['ar_logits'].shape == (15, 20) if ok_has_ar else False
    loss_fn = METALoss(1.0, 0.5, 0.0)
    total, ld = loss_fn(pred, batch, use_ar=True); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_seq_decoder=True, skip_mask_tokens=True, skip_dynamics_decoders=True)
    ok_seq_skip = pred['seq_logits'].abs().sum().item() == 0.0
    passed = ok_has_ar and ok_shape and grads_ok and ok_seq_skip
    report("Full model sched_sampling", passed,
           f"ar={ok_has_ar}, grads={grads_ok}, seq_skip={ok_seq_skip}" +
           (f", missing={no_grad[:3]}" if not grads_ok else ""), time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 15: No higher ranks (graph-only mode)
# ══════════════════════════════════════════════════════════════════════════════
def test_no_higher_ranks():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='attn')
    batch = make_batch(20, 60, 0, 0)
    batch['bend_feat'] = torch.zeros(0, 1); batch['torsion_feat'] = torch.zeros(0, 2)
    batch['biochem_targets'] = torch.zeros(0, 4); batch['torsion_biochem_targets'] = torch.zeros(0, 4)
    for k in ['nbr2_src','nbr2_dst','nbr3_src','nbr3_dst','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']:
        batch[k] = torch.zeros(0, dtype=torch.long)
    batch['bends'] = torch.zeros(0, 3, dtype=torch.long); batch['torsions'] = torch.zeros(0, 4, dtype=torch.long)
    batch['n_bends'] = torch.tensor([0]); batch['n_torsions'] = torch.tensor([0])
    batch['bend_batch'] = torch.zeros(0, dtype=torch.long); batch['torsion_batch'] = torch.zeros(0, dtype=torch.long)
    pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.0); total, ld = loss_fn(pred, batch); total.backward()
    report("No higher ranks (graph-only)", True, f"loss={ld['total']:.4f}", time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 21: Feature masking
# ══════════════════════════════════════════════════════════════════════════════
def test_feature_masking():
    t0 = time.time(); import torch
    from src.model_utils import METAModel
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='conv', mask_ratio=0.3)
    batch = make_batch(20, 60, 40, 15)
    model.train(); pred = model(batch)
    total_masked_train = sum(m.sum().item() for m in pred['recon_masks']); ok_train = total_masked_train > 0
    model.eval()
    with torch.no_grad():
        pred_eval = model(batch)
        total_masked_eval = sum(m.sum().item() for m in pred_eval['recon_masks'])
    ok_eval = total_masked_eval == 0
    report("Feature masking", ok_train and ok_eval,
           f"train_masked={int(total_masked_train)}, eval_masked={int(total_masked_eval)}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 22: No dynamics
# ══════════════════════════════════════════════════════════════════════════════
def test_no_dynamics():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='conv')
    batch = make_batch(20, 60, 40, 15)
    batch['has_dynamics'] = torch.tensor([0]); batch['msf'] = torch.zeros(20); batch['pair_var'] = torch.zeros(60)
    pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.5); total, ld = loss_fn(pred, batch); total.backward()
    ok = ld['msf_loss'] == 0.0 and ld['pair_var_loss'] == 0.0
    report("No dynamics fallback", ok, f"msf={ld['msf_loss']}, pv={ld['pair_var_loss']}", time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 23: Pointer network standalone
# ══════════════════════════════════════════════════════════════════════════════
def test_pointer_standalone():
    t0 = time.time(); import torch
    from src.model_utils import PointerNetwork
    ptr = PointerNetwork(32, 0.0); emb = torch.randn(25, 32)
    perm1, lp1 = ptr(emb, chunk_size=1); ok_perm1 = len(set(perm1.tolist())) == 25
    perm5, lp5 = ptr(emb, chunk_size=5); ok_perm5 = len(set(perm5.tolist())) == 25
    emb2 = torch.randn(10, 32, requires_grad=True)
    _, lp_g = ptr(emb2, chunk_size=1); lp_g.sum().backward(); ok_grad = emb2.grad is not None
    report("PointerNetwork standalone", ok_perm1 and ok_perm5 and ok_grad,
           f"seq={ok_perm1}, chunk={ok_perm5}, grad={ok_grad}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 24: Torsion biochemistry decoder
# ══════════════════════════════════════════════════════════════════════════════
def test_torsion_biochem():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='conv')
    batch = make_batch(20, 60, 40, 15); pred = model(batch)
    ok_bend = pred['biochem_pred'].shape == (40, 4); ok_torsion = pred['torsion_biochem_pred'].shape == (15, 4)
    loss_fn = METALoss(1.0, 0.5, 0.0); total, ld = loss_fn(pred, batch); ok_loss = ld['biochem_loss'] > 0
    total.backward()
    report("Torsion biochemistry", ok_bend and ok_torsion and ok_loss,
           f"bend={pred['biochem_pred'].shape}, torsion={pred['torsion_biochem_pred'].shape}", time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 25: Curriculum phases
# ══════════════════════════════════════════════════════════════════════════════
def test_curriculum_phases():
    t0 = time.time()
    from train_utils import TrainingCurriculum
    cur = TrainingCurriculum(p1=20, p2=30, p3=50, gamma_target=0.5)
    ok_p1 = cur.get_phase(0) == 1 and cur.get_phase(19) == 1
    ok_p2 = cur.get_phase(20) == 2 and cur.get_phase(49) == 2
    ok_p3 = cur.get_phase(50) == 3 and cur.get_phase(99) == 3
    ok_p4 = cur.get_phase(100) == 4 and cur.get_phase(200) == 4
    ok_ar = not cur.use_ar(10) and not cur.use_ar(30) and cur.use_ar(60) and cur.use_ar(110)
    passed = ok_p1 and ok_p2 and ok_p3 and ok_p4 and ok_ar
    report("Curriculum phases", passed, f"P1={ok_p1}, P2={ok_p2}, P3={ok_p3}, P4={ok_p4}, AR={ok_ar}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 26: parse_layer_types
# ══════════════════════════════════════════════════════════════════════════════
def test_parse_layer_types():
    t0 = time.time()
    from src.model_utils import parse_layer_types
    ok1 = parse_layer_types('attn', 4) == [True]*4; ok2 = parse_layer_types('conv', 3) == [False]*3
    ok3 = parse_layer_types('attn,conv,conv', 3) == [True, False, False]
    try: parse_layer_types('attn,conv', 3); ok4 = False
    except ValueError: ok4 = True
    report("parse_layer_types", ok1 and ok2 and ok3 and ok4,
           f"attn={ok1}, conv={ok2}, mixed={ok3}, error={ok4}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 27: MSF and PairwiseVar decoders
# ══════════════════════════════════════════════════════════════════════════════
def test_dynamics_decoders():
    t0 = time.time(); import torch
    from src.model_utils import MSFDecoder, PairwiseVarDecoder
    dm = 32; msf_dec = MSFDecoder(dm); pv_dec = PairwiseVarDecoder(dm)
    h0 = torch.randn(20, dm); h1 = torch.randn(50, dm)
    es = torch.randint(0, 20, (50,)); ed = torch.randint(0, 20, (50,))
    msf = msf_dec(h0); pv = pv_dec(h0, h1, es, ed)
    ok_msf = msf.shape == (20,) and (msf >= 0).all()
    ok_pv = pv.shape == (50,) and (pv >= 0).all()
    report("MSF+PairwiseVar decoders", ok_msf.item() and ok_pv.item(), f"msf={msf.shape}, pv={pv.shape}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 28: E2E gradient flow
# AR active + masking active → seq_decoder unused, mask_tokens DO get grads
# ══════════════════════════════════════════════════════════════════════════════
def test_gradient_flow():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=2, n_layers=2, d_node=23, d_edge=37,
                      layer_types='attn,conv', use_ar=True, use_pointer=True, chunk_size=2,
                      mask_ratio=0.15, topo_mask_ratio=0.1)
    batch = make_batch(20, 60, 40, 15)
    model.train(); pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.5, 0.3, 0.2)
    total, ld = loss_fn(pred, batch, use_ar=True); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_unused_heads=False, skip_seq_decoder=True)
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    detail = f"all {n_params} params OK" if grads_ok else f"missing: {no_grad[:5]}"
    report("E2E gradient flow", grads_ok, detail, time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 29: Batch>1 with AR + pointer
# AR active + mask_ratio=0 → skip seq_decoder + mask_tokens
# ══════════════════════════════════════════════════════════════════════════════
def test_batch_gt1_pointer():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=2, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', use_ar=True, use_pointer=True, chunk_size=2)
    b1 = make_batch(15, 40, 25, 10); b2 = make_batch(10, 30, 20, 8)
    batch = {}
    batch['node_feat'] = torch.cat([b1['node_feat'], b2['node_feat']])
    batch['edge_feat'] = torch.cat([b1['edge_feat'], b2['edge_feat']])
    batch['bend_feat'] = torch.cat([b1['bend_feat'], b2['bend_feat']])
    batch['torsion_feat'] = torch.cat([b1['torsion_feat'], b2['torsion_feat']])
    batch['seq_idx'] = torch.cat([b1['seq_idx'], b2['seq_idx']])
    batch['biochem_targets'] = torch.cat([b1['biochem_targets'], b2['biochem_targets']])
    batch['torsion_biochem_targets'] = torch.cat([b1['torsion_biochem_targets'], b2['torsion_biochem_targets']])
    batch['msf'] = torch.cat([b1['msf'], b2['msf']]); batch['pair_var'] = torch.cat([b1['pair_var'], b2['pair_var']])
    batch['has_dynamics'] = torch.tensor([1, 1])
    batch['edge_src'] = torch.cat([b1['edge_src'], b2['edge_src']+15])
    batch['edge_dst'] = torch.cat([b1['edge_dst'], b2['edge_dst']+15])
    batch['bends'] = torch.cat([b1['bends'], b2['bends']+15])
    batch['torsions'] = torch.cat([b1['torsions'], b2['torsions']+15])
    batch['nbr0_src'] = torch.cat([b1['nbr0_src'], b2['nbr0_src']+15])
    batch['nbr0_dst'] = torch.cat([b1['nbr0_dst'], b2['nbr0_dst']+15])
    batch['nbr1_src'] = torch.cat([b1['nbr1_src'], b2['nbr1_src']+40])
    batch['nbr1_dst'] = torch.cat([b1['nbr1_dst'], b2['nbr1_dst']+40])
    batch['nbr2_src'] = torch.cat([b1['nbr2_src'], b2['nbr2_src']+25])
    batch['nbr2_dst'] = torch.cat([b1['nbr2_dst'], b2['nbr2_dst']+25])
    batch['nbr3_src'] = torch.cat([b1['nbr3_src'], b2['nbr3_src']+10])
    batch['nbr3_dst'] = torch.cat([b1['nbr3_dst'], b2['nbr3_dst']+10])
    batch['inc_01_edge'] = torch.cat([b1['inc_01_edge'], b2['inc_01_edge']+40])
    batch['inc_01_node'] = torch.cat([b1['inc_01_node'], b2['inc_01_node']+15])
    batch['inc_12_bend'] = torch.cat([b1['inc_12_bend'], b2['inc_12_bend']+25])
    batch['inc_12_edge'] = torch.cat([b1['inc_12_edge'], b2['inc_12_edge']+40])
    batch['inc_23_torsion'] = torch.cat([b1['inc_23_torsion'], b2['inc_23_torsion']+10])
    batch['inc_23_bend'] = torch.cat([b1['inc_23_bend'], b2['inc_23_bend']+25])
    batch['n_res'] = torch.tensor([15,10]); batch['n_edges'] = torch.tensor([40,30])
    batch['n_bends'] = torch.tensor([25,20]); batch['n_torsions'] = torch.tensor([10,8])
    batch['node_batch'] = torch.cat([torch.zeros(15,dtype=torch.long), torch.ones(10,dtype=torch.long)])
    batch['edge_batch'] = torch.cat([torch.zeros(40,dtype=torch.long), torch.ones(30,dtype=torch.long)])
    batch['bend_batch'] = torch.cat([torch.zeros(25,dtype=torch.long), torch.ones(20,dtype=torch.long)])
    batch['torsion_batch'] = torch.cat([torch.zeros(10,dtype=torch.long), torch.ones(8,dtype=torch.long)])
    model.train(); pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.5)
    total, ld = loss_fn(pred, batch, use_ar=True); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_seq_decoder=True, skip_mask_tokens=True)
    perm = pred['perm']; ok_shape = perm.shape == (25,)
    perm0 = set(perm[:15].tolist()); perm1 = set(perm[15:].tolist())
    ok_iso = perm0.issubset(set(range(15))) and perm1.issubset(set(range(15,25)))
    ok_v0 = len(perm0) == 15; ok_v1 = len(perm1) == 10
    passed = grads_ok and ok_shape and ok_iso and ok_v0 and ok_v1
    report("Batch>1 pointer isolation", passed,
           f"grads={grads_ok}, iso={ok_iso}, v0={ok_v0}, v1={ok_v1}" +
           (f", missing={no_grad[:3]}" if not grads_ok else ""), time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 30: global_ar mode (per_protein_ar=False)
# AR active + mask_ratio=0 + gamma=0 → skip seq_decoder, mask_tokens, dynamics_decoders
# ══════════════════════════════════════════════════════════════════════════════
def test_global_ar_mode():
    t0 = time.time(); import torch
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=32, n_heads=2, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', use_ar=True, use_pointer=True, chunk_size=2, per_protein_ar=False)
    batch = make_batch(20, 60, 40, 15)
    model.train(); pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.0)
    total, ld = loss_fn(pred, batch, use_ar=True); total.backward()
    grads_ok, no_grad = _check_grads(model, skip_seq_decoder=True, skip_mask_tokens=True, skip_dynamics_decoders=True)
    ok_perm = pred['perm'].shape == (20,) and len(set(pred['perm'].tolist())) == 20
    report("global_ar mode", grads_ok and ok_perm,
           f"grads={grads_ok}, perm={ok_perm}, loss={ld['total']:.4f}" +
           (f", missing={no_grad[:3]}" if not grads_ok else ""), time.time()-t0)
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
# TEST 31: global vs per_protein at batch_size=1
# ══════════════════════════════════════════════════════════════════════════════
def test_global_vs_perprotein_bs1():
    t0 = time.time(); import torch
    from src.model_utils import METAModel
    torch.manual_seed(99)
    model_pp = METAModel(d_model=32, n_heads=2, n_layers=1, d_node=23, d_edge=37,
                         layer_types='conv', use_ar=True, use_pointer=False, per_protein_ar=True)
    torch.manual_seed(99)
    model_gl = METAModel(d_model=32, n_heads=2, n_layers=1, d_node=23, d_edge=37,
                         layer_types='conv', use_ar=True, use_pointer=False, per_protein_ar=False)
    batch = make_batch(15, 40, 25, 10); model_pp.eval(); model_gl.eval()
    with torch.no_grad():
        torch.manual_seed(42); pred_pp = model_pp(batch)
        torch.manual_seed(42); pred_gl = model_gl(batch)
    ok_seq = torch.allclose(pred_pp['seq_logits'], pred_gl['seq_logits'], atol=1e-5)
    ok_msf = torch.allclose(pred_pp['msf_pred'], pred_gl['msf_pred'], atol=1e-5)
    ok_ar = pred_pp['ar_logits'].shape == pred_gl['ar_logits'].shape
    report("global vs per_protein bs=1", ok_seq and ok_msf and ok_ar,
           f"seq={ok_seq}, msf={ok_msf}, ar_shape={ok_ar}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
# TEST 32: SparseARDecoderLayer standalone
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_ar_decoder_layer():
    t0 = time.time(); import torch
    from src.model_utils import SparseARDecoderLayer
    dm, N, E = 32, 20, 60
    layer = SparseARDecoderLayer(dm, dropout=0.0, scale=30)
    h_V = torch.randn(N, dm, requires_grad=True)
    h_ESV = torch.randn(E, 3*dm); edge_dst = torch.randint(0, N, (E,))
    out = layer(h_V, h_ESV, edge_dst, N)
    ok_shape = out.shape == (N, dm)
    out.sum().backward(); ok_grad = h_V.grad is not None
    ok_changed = not torch.allclose(out.detach(), h_V.detach(), atol=1e-6)
    report("SparseARDecoderLayer", ok_shape and ok_grad and ok_changed,
           f"shape={ok_shape}, grad={ok_grad}, residual={ok_changed}", time.time()-t0)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("META v6.2 Comprehensive Test Suite (31 tests)")
    print("SparseARDecoder + scheduled sampling + all layers + batch>1")
    print("=" * 70)
    tests = [
        ("1  Scatter Utilities", test_scatter_utils),
        ("2  SparseNeighbourhoodSelfAttn", test_sparse_self_attn),
        ("3  SparseGraphConv", test_sparse_graph_conv),
        ("4  CrossAttn + IncConv", test_cross_ops),
        ("5  METALayer (4-rank)", test_meta_layer),
        ("6  SparseARDecoder (shape+grad+causal)", test_sparse_ar_decoder),
        ("6b SparseARDecoder causal mask", test_sparse_causal_mask),
        ("6c SparseARDecoder sched sampling", test_sparse_sched_sampling),
        ("6d SparseARDecoder.generate()", test_sparse_ar_generate),
        ("7  REINFORCE pointer", test_reinforce_loss),
        ("8  Chunk annealing", test_chunk_annealing),
        ("9  Scheduled sampling ratio", test_scheduled_sampling),
        ("10 Clean neg sampling", test_clean_neg_sampling),
        ("11 Vectorized pool", test_vectorized_pool),
        ("16 Full model (conv, no AR)", test_full_model_conv),
        ("17 Full model (hybrid, no AR)", test_full_model_hybrid),
        ("18 Full AR+Pointer+Masking", test_full_model_ar_pointer),
        ("19 Full model sched_sampling", test_full_model_sched_sampling),
        ("15 No higher ranks", test_no_higher_ranks),
        ("21 Feature masking", test_feature_masking),
        ("22 No dynamics", test_no_dynamics),
        ("23 Pointer standalone", test_pointer_standalone),
        ("24 Torsion biochem", test_torsion_biochem),
        ("25 Curriculum phases", test_curriculum_phases),
        ("26 parse_layer_types", test_parse_layer_types),
        ("27 Dynamics decoders", test_dynamics_decoders),
        ("28 E2E gradient flow", test_gradient_flow),
        ("29 Batch>1 pointer isolation", test_batch_gt1_pointer),
        ("30 global_ar mode", test_global_ar_mode),
        ("31 global vs per_protein bs=1", test_global_vs_perprotein_bs1),
        ("32 SparseARDecoderLayer", test_sparse_ar_decoder_layer),
    ]
    passed = 0; failed = 0; errors = []
    for i, (name, fn) in enumerate(tests):
        print(f"\n[{i+1}/{len(tests)}] {name}...")
        try:
            fn()
            if results[-1][1]: passed += 1
            else: failed += 1; errors.append(name)
        except Exception as e:
            failed += 1; errors.append(name)
            report(name, False, f"EXCEPTION: {e}")
            traceback.print_exc()
    elapsed = time.time() - t0_global
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed ({elapsed:.1f}s total)")
    if errors: print(f"FAILED: {', '.join(errors)}")
    else: print("ALL TESTS PASSED")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)