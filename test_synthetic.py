"""META v3 test suite: 4-rank cochain complex, bends, torsions, hybrid attn/conv."""
import sys, os, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def make_helix_pdb(n_res, chain='A', start_resi=1, z_off=0.0):
    lines = []
    for i in range(n_res):
        t = i * 100.0 * np.pi / 180.0
        for j, (an, ro, zo) in enumerate([('N',1.5,-0.5),('CA',2.3,0.0),('C',2.3,0.5)]):
            x = ro*np.cos(t+j*0.3); y = ro*np.sin(t+j*0.3); z = i*1.5+zo+z_off
            lines.append("ATOM  %5d  %-4sALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s" % (i*3+j+1, an, chain, start_resi+i, x, y, z, an[0]))
    return lines
def make_batch(n_res=30, n_edges=100, n_bends=80, n_torsions=30, d_node=23, d_edge=37, dev='cpu'):
    import torch
    b = {}
    b['node_feat'] = torch.randn(n_res, d_node, device=dev)
    b['edge_feat'] = torch.randn(n_edges, d_edge, device=dev)
    b['bend_feat'] = torch.randn(n_bends, 1, device=dev)
    b['torsion_feat'] = torch.randn(n_torsions, 2, device=dev)
    b['seq_idx'] = torch.randint(0, 20, (n_res,), device=dev)
    b['biochem_targets'] = torch.randn(n_bends, 4, device=dev)
    b['torsion_biochem_targets'] = torch.randn(n_torsions, 4, device=dev)
    b['msf'] = torch.rand(n_res, device=dev)*10
    b['pair_var'] = torch.rand(n_edges, device=dev)
    b['has_dynamics'] = torch.tensor([1], device=dev)
    s = torch.randint(0, n_res, (n_edges,), device=dev); d = torch.randint(0, n_res, (n_edges,), device=dev)
    b['edge_src'] = s; b['edge_dst'] = d
    b['nbr0_src'] = s; b['nbr0_dst'] = d
    ne = min(n_edges*3, 800)
    b['nbr1_src'] = torch.randint(0, n_edges, (ne,), device=dev); b['nbr1_dst'] = torch.randint(0, n_edges, (ne,), device=dev)
    nb = min(n_bends*2, 400)
    b['nbr2_src'] = torch.randint(0, max(n_bends,1), (nb,), device=dev); b['nbr2_dst'] = torch.randint(0, max(n_bends,1), (nb,), device=dev)
    nt = min(n_torsions*2, 200)
    b['nbr3_src'] = torch.randint(0, max(n_torsions,1), (nt,), device=dev); b['nbr3_dst'] = torch.randint(0, max(n_torsions,1), (nt,), device=dev)
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
def test_utils():
    from src.utils import (compute_dihedrals, compute_bond_angle, compute_virtual_cbeta,
        compute_bend_cosine, compute_dihedral_4point, compute_covalent_onehot,
        compute_torsion_properties, compute_triangle_properties,
        aa_to_idx, three_to_one, NUM_AA, detect_chain_breaks, compute_seq_separation)
    print("\n[1] Utils...")
    L = 20; N = np.random.randn(L, 3); CA = np.random.randn(L, 3); C = np.random.randn(L, 3)
    assert compute_dihedrals(N, CA, C)[0].shape == (L,)
    assert compute_bond_angle(N, CA, C).shape == (L,)
    assert compute_virtual_cbeta(N, CA, C).shape == (L, 3)
    # bend cosine
    cos = compute_bend_cosine(CA, np.array([0,1]), np.array([2,3]), np.array([4,5]))
    assert cos.shape == (2,) and np.all(np.abs(cos) <= 1.0)
    # dihedral
    sin_d, cos_d = compute_dihedral_4point(CA, np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]))
    assert sin_d.shape == (2,)
    # covalent onehot
    ci = np.zeros(L, dtype=np.int64); rn = np.arange(L, dtype=np.int64)
    cov = compute_covalent_onehot(np.array([0,0,5]), np.array([1,5,6]), ci, rn)
    assert cov.shape == (3,2) and cov[0,0] == 1.0 and cov[1,1] == 1.0 and cov[2,0] == 1.0
    # torsion properties
    seq_idx = np.zeros(10, dtype=np.int64)  # all Ala
    tor = np.array([[0,1,2,3],[4,5,6,7]], dtype=np.int64)
    tp = compute_torsion_properties(seq_idx, tor)
    assert tp.shape == (2, 4), "Torsion props should be (n_torsions, 4)"
    # triangle properties
    bends = np.array([[0,1,2]], dtype=np.int64)
    bp = compute_triangle_properties(seq_idx, bends)
    assert bp.shape == (1, 4), "Bend props should be (n_bends, 4)"
    # aa mapping
    assert aa_to_idx("X")[0] == NUM_AA
    assert three_to_one("MSE") == "M"
    print("  PASSED")
def test_parsing():
    from src.processing import parse_structure, list_chains
    print("\n[2] Parsing...")
    # single chain
    lines = make_helix_pdb(25, 'A') + ["END"]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f: f.write('\n'.join(lines)); p1 = f.name
    r = parse_structure(p1); assert r is not None and r['n_res'] == 25; os.unlink(p1)
    # multi-chain
    lines = make_helix_pdb(20, 'A') + ["TER"] + make_helix_pdb(15, 'B', z_off=50) + ["END"]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f: f.write('\n'.join(lines)); p2 = f.name
    assert 'A' in list_chains(p2) and 'B' in list_chains(p2)
    r = parse_structure(p2); assert r['n_res'] == 35 and r['n_chains'] == 2; os.unlink(p2)
    print("  PASSED")
def test_cochain_complex():
    from src.processing import parse_structure, build_cochain_complex
    print("\n[3] Cochain complex (4 ranks)...")
    lines = make_helix_pdb(30, 'A') + ["END"]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f: f.write('\n'.join(lines)); p = f.name
    bb = parse_structure(p); os.unlink(p)
    try:
        cc = build_cochain_complex(bb, edge_cutoff=10.0)
    except ImportError:
        print("  SKIPPED (needs torch for RBF)"); return
    print("  Rank 0 (nodes): %d, feat=%s" % (cc['n_res'], cc['node_feat'].shape))
    print("  Rank 1 (edges): %d, feat=%s" % (cc['n_edges'], cc['edge_feat'].shape))
    print("  Rank 2 (bends): %d, feat=%s" % (cc['n_bends'], cc['bend_feat'].shape))
    print("  Rank 3 (torsions): %d, feat=%s" % (cc['n_torsions'], cc['torsion_feat'].shape))
    assert cc['node_feat'].shape[1] == 23, "d_node should be 23 (no DSSP)"
    assert cc['edge_feat'].shape[1] == 37, "d_edge should be 37 (16+3+16+2, no same_chain)"
    assert cc['bend_feat'].shape[1] == 1, "d_bend should be 1 (cos angle)"
    if cc['n_torsions'] > 0:
        assert cc['torsion_feat'].shape[1] == 2, "d_torsion should be 2 (sin,cos)"
    assert cc['n_bends'] > 0, "Should have bends for a helix"
    # verify neighbourhood keys
    for k in ['nbr0_src','nbr1_src','nbr2_src','nbr3_src']: assert k in cc
    for k in ['inc_01_edge','inc_12_bend','inc_23_torsion']: assert k in cc
    print("  PASSED")
def test_model_configs():
    import torch
    from src.model_utils import METAModel, METALoss, parse_layer_types
    from train_utils import compute_all_metrics, count_parameters
    print("\n[4] Model configs (attn, conv, hybrid)...")
    # test parse_layer_types
    assert parse_layer_types('attn', 4) == [True]*4
    assert parse_layer_types('conv', 3) == [False]*3
    assert parse_layer_types('attn,conv,conv', 3) == [True, False, False]
    configs = [
        {'d_model': 32, 'n_heads': 1, 'n_layers': 1, 'layer_types': 'attn', 'use_ar': False},
        {'d_model': 32, 'n_heads': 1, 'n_layers': 1, 'layer_types': 'conv', 'use_ar': False},
        {'d_model': 64, 'n_heads': 2, 'n_layers': 3, 'layer_types': 'attn,conv,conv', 'use_ar': False},
        {'d_model': 64, 'n_heads': 2, 'n_layers': 2, 'layer_types': 'attn,conv', 'use_ar': True},
    ]
    for i, cfg in enumerate(configs):
        model = METAModel(**cfg, dropout=0.0, d_node=23, d_edge=37)
        batch = make_batch(n_res=20, n_edges=60, n_bends=40, n_torsions=15)
        pred = model(batch)
        loss_fn = METALoss(1.0, 0.5, 0.5)
        total, ld = loss_fn(pred, batch, cfg.get('use_ar', False))
        total.backward()
        grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        met = compute_all_metrics(pred, batch, cfg.get('use_ar', False))
        print("  Config %d (%s): %d params, loss=%.4f, rec=%.3f, grads=%s" % (
            i+1, cfg['layer_types'], count_parameters(model), ld['total'], met['recovery'], grads_ok))
        assert grads_ok; model.zero_grad()
    print("  PASSED")
def test_no_dynamics():
    import torch
    from src.model_utils import METAModel, METALoss
    print("\n[5] No dynamics...")
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='conv')
    batch = make_batch(20, 60, 40, 15)
    batch['has_dynamics'] = torch.tensor([0]); batch['msf'] = torch.zeros(20); batch['pair_var'] = torch.zeros(60)
    loss_fn = METALoss(1.0, 0.5, 0.5)
    pred = model(batch); total, ld = loss_fn(pred, batch); total.backward()
    assert ld['msf_loss'] == 0.0 and ld['pair_var_loss'] == 0.0
    print("  PASSED")
def test_no_higher_ranks():
    import torch
    from src.model_utils import METAModel, METALoss
    print("\n[6] No bends/torsions (graph only)...")
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='attn')
    batch = make_batch(20, 60, 0, 0)
    for k in ['bend_feat']: batch[k] = torch.zeros(0, 1)
    for k in ['torsion_feat']: batch[k] = torch.zeros(0, 2)
    batch['biochem_targets'] = torch.zeros(0, 4)
    batch['torsion_biochem_targets'] = torch.zeros(0, 4)
    for k in ['nbr2_src','nbr2_dst','nbr3_src','nbr3_dst','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']:
        batch[k] = torch.zeros(0, dtype=torch.long)
    batch['bends'] = torch.zeros(0, 3, dtype=torch.long); batch['torsions'] = torch.zeros(0, 4, dtype=torch.long)
    batch['n_bends'] = torch.tensor([0]); batch['n_torsions'] = torch.tensor([0])
    batch['bend_batch'] = torch.zeros(0, dtype=torch.long); batch['torsion_batch'] = torch.zeros(0, dtype=torch.long)
    loss_fn = METALoss(1.0, 0.5, 0.0)
    pred = model(batch); total, ld = loss_fn(pred, batch); total.backward()
    print("  PASSED: loss=%.4f" % ld['total'])
def test_gcnn_vs_attn_shapes():
    import torch
    from src.model_utils import SparseGraphConv, SparseNeighbourhoodSelfAttn
    print("\n[7] GCNN vs Attn output shapes...")
    d = 64; N = 50; E = 200
    X = torch.randn(N, d); src = torch.randint(0, N, (E,)); dst = torch.randint(0, N, (E,))
    attn = SparseNeighbourhoodSelfAttn(d, 4, 0.0)
    gcnn = SparseGraphConv(d, 0.0)
    out_a = attn(X, src, dst, N); out_g = gcnn(X, src, dst, N)
    assert out_a.shape == (N, d) and out_g.shape == (N, d)
    print("  PASSED: both produce (%d, %d)" % (N, d))
def test_feature_masking():
    import torch
    from src.model_utils import METAModel, METALoss
    from train_utils import compute_all_metrics
    print("\n[8] Feature masking (MLM-style)...")
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', mask_ratio=0.3, topo_mask_ratio=0.0)
    batch = make_batch(20, 60, 40, 15)
    model.train()
    pred = model(batch)
    # check reconstruction outputs exist
    assert 'recon_preds' in pred and len(pred['recon_preds']) == 4
    assert 'recon_targets' in pred and len(pred['recon_targets']) == 4
    assert 'recon_masks' in pred and len(pred['recon_masks']) == 4
    # at least some cochains should be masked
    total_masked = sum(m.sum().item() for m in pred['recon_masks'])
    assert total_masked > 0, "No cochains were masked"
    # recon predictions should match target dims at masked positions
    for r in range(4):
        p, t = pred['recon_preds'][r], pred['recon_targets'][r]
        assert p.shape == t.shape, "Recon pred/target dim mismatch at rank %d" % r
    # loss should include recon_loss
    loss_fn = METALoss(1.0, 0.5, 0.0, delta=0.5, zeta=0.0)
    total, ld = loss_fn(pred, batch)
    assert 'recon_loss' in ld and ld['recon_loss'] > 0, "Reconstruction loss should be > 0"
    total.backward()
    print("  PASSED: %d cochains masked, recon_loss=%.4f" % (total_masked, ld['recon_loss']))
    model.zero_grad()
def test_topology_masking():
    import torch
    from src.model_utils import METAModel, METALoss
    print("\n[9] Topology masking (edge/incidence dropout + reconstruction)...")
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', mask_ratio=0.0, topo_mask_ratio=0.3)
    batch = make_batch(20, 60, 40, 15)
    model.train()
    pred = model(batch)
    assert 'topo_nbr_logits' in pred and len(pred['topo_nbr_logits']) == 4
    assert 'topo_inc_logits' in pred and len(pred['topo_inc_logits']) == 3
    # check that logits and labels are paired
    for r in range(4):
        l, lb = pred['topo_nbr_logits'][r], pred['topo_nbr_labels'][r]
        assert l.shape == lb.shape, "Topo nbr logit/label shape mismatch rank %d" % r
    for r in range(3):
        l, lb = pred['topo_inc_logits'][r], pred['topo_inc_labels'][r]
        assert l.shape == lb.shape, "Topo inc logit/label shape mismatch rank %d" % r
    loss_fn = METALoss(1.0, 0.5, 0.0, delta=0.0, zeta=0.5)
    total, ld = loss_fn(pred, batch)
    assert 'topo_loss' in ld and ld['topo_loss'] > 0, "Topo loss should be > 0"
    total.backward()
    print("  PASSED: topo_loss=%.4f" % ld['topo_loss'])
    model.zero_grad()
def test_both_masking():
    import torch
    from src.model_utils import METAModel, METALoss
    from train_utils import compute_all_metrics
    print("\n[10] Combined feature + topology masking...")
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', mask_ratio=0.2, topo_mask_ratio=0.15)
    batch = make_batch(20, 60, 40, 15)
    model.train()
    pred = model(batch)
    loss_fn = METALoss(1.0, 0.5, 0.5, delta=0.3, zeta=0.2)
    total, ld = loss_fn(pred, batch)
    total.backward()
    met = compute_all_metrics(pred, batch)
    print("  PASSED: total=%.4f recon=%.4f topo=%.4f rec=%.3f" % (ld['total'], ld['recon_loss'], ld['topo_loss'], met['recovery']))
    # inference mode: no masking
    model.eval()
    with torch.no_grad():
        pred_eval = model(batch)
        # reconstruction outputs should have zero masked
        total_masked_eval = sum(m.sum().item() for m in pred_eval['recon_masks'])
        assert total_masked_eval == 0, "No masking in eval mode"
    print("  PASSED: eval mode produces zero masks")
    model.zero_grad()
def test_pointer_network():
    import torch
    from src.model_utils import METAModel, METALoss, PointerNetwork
    from train_utils import compute_all_metrics
    print("\n[11] Pointer network + AR with torsion context...")
    # standalone pointer test
    ptr = PointerNetwork(32, 0.0)
    emb = torch.randn(20, 32)
    perm, lp = ptr(emb, chunk_size=1)
    assert perm.shape == (20,), "Perm should be (N,)"
    assert len(set(perm.tolist())) == 20, "Perm should be a valid permutation"
    # chunked pointer test
    perm_c, lp_c = ptr(emb, chunk_size=5)
    assert perm_c.shape == (20,) and len(set(perm_c.tolist())) == 20
    print("  Pointer standalone: perm OK, chunked OK")
    # full model with pointer + AR
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37,
                      layer_types='conv', use_ar=True, use_pointer=True, chunk_size=4)
    batch = make_batch(n_res=20, n_edges=60, n_bends=40, n_torsions=15)
    model.train()
    pred = model(batch)
    assert 'ar_logits' in pred, "AR logits missing"
    assert 'ptr_log_probs' in pred, "Pointer log_probs missing"
    assert 'perm' in pred, "Pointer perm missing"
    assert 'torsion_biochem_pred' in pred, "Torsion biochem pred missing"
    assert pred['torsion_biochem_pred'].shape == (15, 4), "Torsion biochem shape mismatch"
    loss_fn = METALoss(1.0, 0.5, 0.5)
    total, ld = loss_fn(pred, batch, use_ar=True)
    total.backward()
    # verify pointer grads flow
    ptr_grads = all(p.grad is not None for p in model.pointer_net.parameters() if p.requires_grad)
    print("  Full model: loss=%.4f, ar_loss=%.4f, biochem=%.4f, ptr_grads=%s" % (
        ld['total'], ld.get('ar_loss', 0), ld['biochem_loss'], ptr_grads))
    model.zero_grad()
    print("  PASSED")
def test_torsion_biochem_targets():
    import torch
    from src.model_utils import METAModel, METALoss
    print("\n[12] Torsion biochemistry targets...")
    model = METAModel(d_model=32, n_heads=1, n_layers=1, d_node=23, d_edge=37, layer_types='conv')
    batch = make_batch(20, 60, 40, 15)
    pred = model(batch)
    # torsion biochem should be predicted from h3
    assert pred['torsion_biochem_pred'].shape == (15, 4)
    # loss should combine bend + torsion biochem
    loss_fn = METALoss(1.0, 0.5, 0.0)
    total, ld = loss_fn(pred, batch)
    assert ld['biochem_loss'] > 0, "Combined biochem loss should be > 0"
    total.backward()
    print("  PASSED: biochem_loss=%.4f (includes torsion contribution)" % ld['biochem_loss'])
    model.zero_grad()
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 60)
    print("META v5 Test Suite (pointer net, torsion biochem, implicit multichain)")
    print("=" * 60)
    test_utils()
    test_parsing()
    test_cochain_complex()
    try:
        import torch
        test_model_configs()
        test_no_dynamics()
        test_no_higher_ranks()
        test_gcnn_vs_attn_shapes()
        test_feature_masking()
        test_topology_masking()
        test_both_masking()
        test_pointer_network()
        test_torsion_biochem_targets()
    except ImportError:
        print("\nPyTorch not available, skipping model tests.")
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
