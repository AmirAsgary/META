"""META processing v6: LMDB streaming, optional dynamics, vectorized ANM+neighbourhood.
PDB/CIF parsing, cochain complex (ranks 0-3), adjacency/coadjacency/incidence.
Cochains: 0=residues, 1=edges, 2=bends(angle), 3=torsions(dihedral)."""
import numpy as np, os, glob, json, logging, warnings, io, pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils import (compute_dihedrals, compute_bond_angle, compute_virtual_cbeta, compute_local_frame,
    project_to_local_frame, rbf_encode, sinusoidal_encode, compute_sasa_shrake_rupley,
    compute_triangle_properties, compute_torsion_properties, compute_bend_cosine, compute_dihedral_4point,
    compute_covalent_onehot, compute_seq_separation, compute_edges_kdtree,
    aa_to_idx, save_features, load_features, three_to_one, THREE_TO_ONE, NUM_AA)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
# ══════════════════════════════════════════════════════════════════════════════
# PDB/CIF Parser
# ══════════════════════════════════════════════════════════════════════════════
def _parse_pdb_lines(pdb_path):
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'): break
            if not line.startswith(('ATOM', 'HETATM')) or len(line) < 54: continue
            aname = line[12:16].strip()
            if aname not in ('N', 'CA', 'C'): continue
            alt = line[16]
            if alt not in (' ', 'A', ''): continue
            try:
                atoms.append({'atom': aname, 'chain': line[21], 'resname': line[17:20].strip(),
                    'resi': line[22:26].strip(), 'icode': line[26] if len(line) > 26 else ' ',
                    'x': float(line[30:38]), 'y': float(line[38:46]), 'z': float(line[46:54])})
            except (ValueError, IndexError): continue
    return atoms
def _parse_cif_lines(cif_path):
    atoms = []; in_atom = False; cm = {}
    with open(cif_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('_atom_site.'):
                in_atom = True; cm[line.split('.')[1].strip()] = len(cm); continue
            if in_atom and (line.startswith('#') or line.startswith('loop_') or (line.startswith('_') and not line.startswith('_atom_site'))):
                if atoms: break
                in_atom = line.startswith('_atom_site.'); continue
            if not in_atom or not line or line.startswith('#'): continue
            p = line.split()
            if len(p) < len(cm): continue
            try:
                g = p[cm.get('group_PDB', 0)]
                if g not in ('ATOM', 'HETATM'): continue
                an = p[cm.get('label_atom_id', cm.get('auth_atom_id', 1))].strip('"')
                if an not in ('N', 'CA', 'C'): continue
                alt = p[cm.get('label_alt_id', 4)] if 'label_alt_id' in cm else '.'
                if alt not in ('.', '?', 'A', ''): continue
                mn = p[cm.get('pdbx_PDB_model_num', -1)] if 'pdbx_PDB_model_num' in cm else '1'
                if mn != '1': continue
                ic = p[cm.get('pdbx_PDB_ins_code', -1)] if 'pdbx_PDB_ins_code' in cm else ' '
                if ic in ('?', '.'): ic = ' '
                atoms.append({'atom': an, 'chain': p[cm.get('auth_asym_id', cm.get('label_asym_id', 6))].strip('"'),
                    'resname': p[cm.get('label_comp_id', cm.get('auth_comp_id', 5))].strip('"'),
                    'resi': p[cm.get('auth_seq_id', cm.get('label_seq_id', 8))].strip('"'), 'icode': ic.strip('"'),
                    'x': float(p[cm.get('Cartn_x', 10)]), 'y': float(p[cm.get('Cartn_y', 11)]), 'z': float(p[cm.get('Cartn_z', 12)])})
            except (ValueError, IndexError, KeyError): continue
    return atoms
def parse_structure(file_path, chain_ids=None, min_len=10):
    ext = Path(file_path).suffix.lower()
    recs = _parse_cif_lines(file_path) if ext in ('.cif', '.mmcif') else _parse_pdb_lines(file_path)
    if not recs: return None
    if isinstance(chain_ids, str): chain_ids = [chain_ids]
    Nm, CAm, Cm, rnm = {}, {}, {}, {}; seen = set()
    for a in recs:
        if chain_ids is not None and a['chain'] not in chain_ids: continue
        key = (a['chain'], a['resi'], a['icode'])
        if (key, a['atom']) in seen: continue
        seen.add((key, a['atom'])); co = np.array([a['x'], a['y'], a['z']])
        if a['atom'] == 'N': Nm[key] = co
        elif a['atom'] == 'CA': CAm[key] = co
        elif a['atom'] == 'C': Cm[key] = co
        if key not in rnm: rnm[key] = a['resname']
    ckeys = [k for k in CAm if k in Nm and k in Cm]
    if len(ckeys) < min_len: return None
    def sk(k):
        try: return (k[0], int(k[1]), k[2])
        except ValueError: return (k[0], 0, k[2])
    ckeys = sorted(ckeys, key=sk)
    N_a = np.array([Nm[k] for k in ckeys]); CA_a = np.array([CAm[k] for k in ckeys]); C_a = np.array([Cm[k] for k in ckeys])
    seq = ''.join(three_to_one(rnm.get(k, 'UNK')) for k in ckeys)
    uc = sorted(set(k[0] for k in ckeys)); cl2i = {c: i for i, c in enumerate(uc)}
    ci = np.array([cl2i[k[0]] for k in ckeys], dtype=np.int64)
    rn = np.array([int(k[1]) if k[1].lstrip('-').isdigit() else 0 for k in ckeys], dtype=np.int64)
    return {'N': N_a, 'CA': CA_a, 'C': C_a, 'seq': seq, 'chain_idx': ci, 'chain_ids_unique': uc,
        'chain_labels': [k[0] for k in ckeys], 'res_numbers': rn, 'res_ids': ckeys,
        'path': file_path, 'n_res': len(ckeys), 'n_chains': len(uc)}
parse_pdb_backbone = parse_structure
def list_chains(file_path):
    ext = Path(file_path).suffix.lower()
    a = _parse_cif_lines(file_path) if ext in ('.cif', '.mmcif') else _parse_pdb_lines(file_path)
    return sorted(set(r['chain'] for r in a))
# ══════════════════════════════════════════════════════════════════════════════
# Cochain Complex Construction (ranks 0-3)
# FIX #11: vectorized neighbourhood construction
# ══════════════════════════════════════════════════════════════════════════════
def _build_pairwise_nbr(member_lists, n_items):
    """Vectorized: given groups of item indices, build all (a,b) pairs within each group.
    member_lists: list of arrays, each containing item indices that share a face.
    Returns (src, dst) arrays."""
    src_parts, dst_parts = [], []
    for grp in member_lists:
        if len(grp) < 2: continue
        g = np.array(grp, dtype=np.int64)
        # cartesian product minus diagonal
        a = np.repeat(g, len(g)); b = np.tile(g, len(g))
        mask = a != b; src_parts.append(a[mask]); dst_parts.append(b[mask])
    if not src_parts: return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(src_parts), np.concatenate(dst_parts)
def _unique_pairs(src, dst):
    """Deduplicate (src, dst) pairs."""
    if len(src) == 0: return src, dst
    pairs = np.stack([src, dst], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
def build_cochain_complex(backbone: Dict, edge_cutoff: float = 8.0) -> Dict:
    """Build cochain complex from backbone. Ranks 0-3 with vectorized neighbourhood."""
    import torch as _t
    N, CA, C, seq = backbone['N'], backbone['CA'], backbone['C'], backbone['seq']
    L = len(CA)
    ci = backbone.get('chain_idx', np.zeros(L, dtype=np.int64))
    rn = backbone.get('res_numbers', np.arange(L, dtype=np.int64))
    seq_idx = aa_to_idx(seq)
    # ── Rank 0: node features (NO DSSP) ──
    phi, psi, omega = compute_dihedrals(N, CA, C, chain_idx=ci)
    phi[np.isnan(phi)] = 0; psi[np.isnan(psi)] = 0; omega[np.isnan(omega)] = 0
    dih = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi), np.sin(omega), np.cos(omega)], -1)
    ba_rbf = rbf_encode(_t.tensor(compute_bond_angle(N, CA, C), dtype=_t.float32), 1.5, 2.5, 16).numpy()
    sasa = compute_sasa_shrake_rupley(CA)[:, None]
    node_feat = np.concatenate([dih, ba_rbf, sasa], -1).astype(np.float32)
    d_node = node_feat.shape[1]
    # ── Rank 1: edge construction ──
    CB = compute_virtual_cbeta(N, CA, C)
    esrc, edst = compute_edges_kdtree(CB, edge_cutoff) if L > 50 else _dense_edges(CB, edge_cutoff)
    n_edges = len(esrc)
    if n_edges == 0:
        esrc, edst = compute_edges_kdtree(CA, edge_cutoff + 2.0) if L > 50 else _dense_edges(CA, edge_cutoff + 2.0)
        n_edges = len(esrc)
    # edge features: dist_rbf(16) + local_dir(3) + seq_sep(16) + covalent(2) = 37
    cb_d = np.linalg.norm(CB[edst] - CB[esrc], axis=-1)
    dist_rbf = rbf_encode(_t.tensor(cb_d, dtype=_t.float32), 0, 20, 16).numpy()
    dir_raw = CA[edst] - CA[esrc]
    dir_n = dir_raw / (np.linalg.norm(dir_raw, axis=-1, keepdims=True) + 1e-8)
    frames = compute_local_frame(N, CA, C)
    loc_dir = project_to_local_frame(dir_n, frames, esrc)
    sep_enc = sinusoidal_encode(_t.tensor(compute_seq_separation(esrc, edst, ci, rn), dtype=_t.float32), 16).numpy()
    cov_oh = compute_covalent_onehot(esrc, edst, ci, rn)
    edge_feat = np.concatenate([dist_rbf, loc_dir, sep_enc, cov_oh], -1).astype(np.float32)
    d_edge = edge_feat.shape[1]
    e2i = {}
    for ei in range(n_edges): e2i[(int(esrc[ei]), int(edst[ei]))] = ei
    adj = defaultdict(set)
    for i, j in zip(esrc.tolist(), edst.tolist()): adj[i].add(j)
    # ── Rank 2: bends (i,j,k) with j central, i<k ──
    bends = []
    for j in range(L):
        nbrs = sorted(adj[j])
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                bends.append((nbrs[ai], j, nbrs[bi]))
    bends = np.array(bends, dtype=np.int64) if bends else np.zeros((0, 3), dtype=np.int64)
    n_bends = len(bends)
    if n_bends > 0:
        bend_feat = compute_bend_cosine(CA, bends[:, 0], bends[:, 1], bends[:, 2])[:, None]
    else:
        bend_feat = np.zeros((0, 1), dtype=np.float32)
    bend_to_edges = []
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        for e in [(i, j), (j, i), (j, k), (k, j)]:
            if e in e2i: bend_to_edges.append((bi, e2i[e]))
    # ── Rank 3: torsions from two bends sharing an edge ──
    edge_key_to_bends = defaultdict(list)
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        edge_key_to_bends[(min(i,j), max(i,j))].append(bi)
        edge_key_to_bends[(min(j,k), max(j,k))].append(bi)
    torsions = []; torsion_bend_pairs = []
    for ekey, bindices in edge_key_to_bends.items():
        if len(bindices) < 2: continue
        for ai in range(len(bindices)):
            for bi_idx in range(ai + 1, len(bindices)):
                b1, b2 = bindices[ai], bindices[bi_idx]
                nodes1 = set(bends[b1].tolist()); nodes2 = set(bends[b2].tolist())
                shared = nodes1 & nodes2
                if len(shared) != 2: continue
                all_nodes = nodes1 | nodes2
                if len(all_nodes) != 4: continue
                j_node, k_node = sorted(shared)
                outer1 = (nodes1 - shared).pop(); outer2 = (nodes2 - shared).pop()
                torsions.append((outer1, j_node, k_node, outer2)); torsion_bend_pairs.append((b1, b2))
    torsions = np.array(torsions, dtype=np.int64) if torsions else np.zeros((0, 4), dtype=np.int64)
    n_torsions = len(torsions)
    if n_torsions > 0:
        sin_d, cos_d = compute_dihedral_4point(CA, torsions[:, 0], torsions[:, 1], torsions[:, 2], torsions[:, 3])
        torsion_feat = np.stack([sin_d, cos_d], -1)
    else:
        torsion_feat = np.zeros((0, 2), dtype=np.float32)
    # ── FIX #11: Vectorized Neighbourhood construction ──
    # Rank 0: adj = edge graph
    nbr0_src, nbr0_dst = esrc.copy(), edst.copy()
    # Rank 1: adj = share a node; coadj = both in same bend
    node2edges = defaultdict(list)
    for ei in range(n_edges): node2edges[int(esrc[ei])].append(ei); node2edges[int(edst[ei])].append(ei)
    eadj_s, eadj_d = _build_pairwise_nbr(list(node2edges.values()), n_edges)
    # edge coadjacency via bends
    bend2edges_map = defaultdict(list)
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        be = set()
        for e in [(i,j),(j,i),(j,k),(k,j)]:
            if e in e2i: be.add(e2i[e])
        bend2edges_map[bi] = list(be)
    ecoadj_s, ecoadj_d = _build_pairwise_nbr(list(bend2edges_map.values()), n_edges)
    nbr1_src, nbr1_dst = _unique_pairs(np.concatenate([eadj_s, ecoadj_s]) if len(eadj_s) else ecoadj_s,
                                         np.concatenate([eadj_d, ecoadj_d]) if len(eadj_d) else ecoadj_d)
    # Rank 2: adj = share an edge; coadj = both in same torsion
    ekey_to_bends_list = defaultdict(list)
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        for e in [(i,j),(j,i),(j,k),(k,j)]:
            if e in e2i: ekey_to_bends_list[(min(e), max(e))].append(bi)
    badj_s, badj_d = _build_pairwise_nbr(list(ekey_to_bends_list.values()), n_bends)
    # bend coadjacency via torsions
    tor2bends_map = defaultdict(list)
    for ti in range(n_torsions):
        b1, b2 = torsion_bend_pairs[ti]; tor2bends_map[ti] = [b1, b2]
    bcoadj_s, bcoadj_d = _build_pairwise_nbr(list(tor2bends_map.values()), n_bends)
    nbr2_src, nbr2_dst = _unique_pairs(np.concatenate([badj_s, bcoadj_s]) if len(badj_s) else bcoadj_s,
                                         np.concatenate([badj_d, bcoadj_d]) if len(badj_d) else bcoadj_d)
    # Rank 3: adj = share a bend; no coadjacency
    tbp_arr = np.array(torsion_bend_pairs, dtype=np.int64) if torsion_bend_pairs else np.zeros((0, 2), dtype=np.int64)
    bend_to_torsions = defaultdict(list)
    for ti in range(n_torsions):
        for bi in tbp_arr[ti]: bend_to_torsions[int(bi)].append(ti)
    tadj_s, tadj_d = _build_pairwise_nbr(list(bend_to_torsions.values()), n_torsions)
    nbr3_src, nbr3_dst = _unique_pairs(tadj_s, tadj_d)
    # ── Incidence matrices ──
    inc_01_edge = np.concatenate([np.arange(n_edges), np.arange(n_edges)]).astype(np.int64)
    inc_01_node = np.concatenate([esrc, edst]).astype(np.int64)
    if bend_to_edges:
        bte = np.unique(np.array(bend_to_edges, dtype=np.int64), axis=0)
        inc_12_bend, inc_12_edge = bte[:, 0], bte[:, 1]
    else:
        inc_12_bend = inc_12_edge = np.zeros(0, dtype=np.int64)
    if n_torsions > 0:
        i23 = []; 
        for ti in range(n_torsions):
            b1, b2 = torsion_bend_pairs[ti]; i23.append((ti, b1)); i23.append((ti, b2))
        i23 = np.unique(np.array(i23, dtype=np.int64), axis=0)
        inc_23_torsion, inc_23_bend = i23[:, 0], i23[:, 1]
    else:
        inc_23_torsion = inc_23_bend = np.zeros(0, dtype=np.int64)
    biochem_targets = compute_triangle_properties(seq_idx, bends) if n_bends > 0 else np.zeros((0, 4), dtype=np.float32)
    torsion_biochem_targets = compute_torsion_properties(seq_idx, torsions) if n_torsions > 0 else np.zeros((0, 4), dtype=np.float32)
    return {'n_res': L, 'n_edges': n_edges, 'n_bends': n_bends, 'n_torsions': n_torsions,
        'd_node': d_node, 'd_edge': d_edge, 'seq_idx': seq_idx, 'seq': np.array(list(seq)),
        'node_feat': node_feat, 'edge_feat': edge_feat, 'bend_feat': bend_feat, 'torsion_feat': torsion_feat,
        'edge_src': esrc, 'edge_dst': edst, 'bends': bends, 'torsions': torsions,
        'chain_idx': ci, 'CA': CA.astype(np.float32),
        'nbr0_src': nbr0_src, 'nbr0_dst': nbr0_dst, 'nbr1_src': nbr1_src, 'nbr1_dst': nbr1_dst,
        'nbr2_src': nbr2_src, 'nbr2_dst': nbr2_dst, 'nbr3_src': nbr3_src, 'nbr3_dst': nbr3_dst,
        'inc_01_edge': inc_01_edge, 'inc_01_node': inc_01_node,
        'inc_12_bend': inc_12_bend, 'inc_12_edge': inc_12_edge,
        'inc_23_torsion': inc_23_torsion, 'inc_23_bend': inc_23_bend,
        'biochem_targets': biochem_targets, 'torsion_biochem_targets': torsion_biochem_targets}
build_combinatorial_complex = build_cochain_complex
def _dense_edges(coords, cutoff):
    L = len(coords); d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    m = (d < cutoff) & (~np.eye(L, dtype=bool)); s, t = np.where(m)
    return s.astype(np.int64), t.astype(np.int64)
# ══════════════════════════════════════════════════════════════════════════════
# ProDy ANM Dynamics — OPTIONAL, vectorized (FIX #9, #10, #7)
# ══════════════════════════════════════════════════════════════════════════════
def compute_anm_dynamics(backbone, n_modes=20, n_conformers=1000, cutoff=15.0, temperature=300.0):
    """Optional ANM dynamics. Returns None if ProDy unavailable."""
    try: import prody; prody.confProDy(verbosity='none')
    except ImportError: logger.warning("ProDy not installed — skipping dynamics"); return None
    CA = backbone['CA']; L = len(CA)
    anm = prody.ANM('protein'); anm.buildHessian(CA, cutoff=cutoff)
    try: anm.calcModes(n_modes=min(n_modes, 3*L-7))
    except Exception: return None
    ev = anm.getEigvals(); evc = anm.getEigvecs(); K = len(ev)
    kBT = 1.38064852e-23 * temperature; g = 1.0
    # vectorized MSF: sum_m (u_m^2 / lambda_m) for each residue
    evc_3d = evc.reshape(L, 3, K)  # (L, 3, K)
    msf = (kBT / g) * np.sum(np.sum(evc_3d ** 2, axis=1) / ev[None, :], axis=1).astype(np.float32)
    # FIX #9: vectorized conformer generation — single matmul instead of M*K loops
    np.random.seed(42)
    amplitudes = np.random.normal(0, 1, (n_conformers, K)) * np.sqrt(kBT / (g * ev))[None, :]  # (M, K)
    displacements = amplitudes @ evc.T  # (M, 3N) -> reshape to (M, L, 3)
    conf = (CA[None, :, :] + displacements.reshape(n_conformers, L, 3)).astype(np.float32)
    # medoid via centered RMSD on subsample
    sub_n = min(200, n_conformers); si = np.random.choice(n_conformers, sub_n, replace=False)
    flat = (conf[si] - conf[si].mean(1, keepdims=True)).reshape(sub_n, -1)
    # vectorized pairwise distance
    ad = np.sqrt(((flat[:, None, :] - flat[None, :, :]) ** 2).mean(-1)).mean(-1)
    mi = si[np.argmin(ad)]
    msf_emp = np.mean(np.sum((conf - conf.mean(0)[None]) ** 2, -1), 0).astype(np.float32)
    return {'msf': msf, 'msf_empirical': msf_emp, 'medoid_coords': conf[mi],
        'medoid_idx': int(mi), 'eigenvalues': ev.astype(np.float32),
        'eigenvectors': evc.astype(np.float32), 'n_modes': K}
def compute_pairwise_dist_var_vectorized(backbone, dynamics, esrc, edst, n_conformers=500, temperature=300.0):
    """FIX #7+#10: reuse cached eigendecomposition, fully vectorized.
    FIX BUG3: displacements are from original equilibrium CA, not medoid."""
    ev = dynamics['eigenvalues']; evc = dynamics['eigenvectors']
    CA = backbone['CA']  # original equilibrium, NOT medoid
    L = len(CA); K = len(ev)
    kBT = 1.38064852e-23 * temperature; g = 1.0
    ns = min(n_conformers, 500)
    np.random.seed(42)
    amps = np.random.normal(0, 1, (ns, K)) * np.sqrt(kBT / (g * ev))[None, :]
    disp = amps @ evc.T  # (ns, 3L)
    coords = CA[None, :, :] + disp.reshape(ns, L, 3)  # (ns, L, 3)
    dists = np.linalg.norm(coords[:, edst] - coords[:, esrc], axis=-1)  # (ns, E)
    return np.var(dists, axis=0).astype(np.float32)
# ══════════════════════════════════════════════════════════════════════════════
# Processing Pipeline — dynamics is OPTIONAL (skip with compute_dynamics=False)
# ══════════════════════════════════════════════════════════════════════════════
def process_single_structure(file_path, cache_dir, chain_ids=None, edge_cutoff=8.0,
                             compute_dynamics=False, n_modes=20, n_conformers=1000,
                             min_len=30, max_len=500):
    """Process one PDB/CIF. compute_dynamics=False by default for fast testing."""
    cs = ""
    if isinstance(chain_ids, str): cs = f"_{chain_ids}"
    elif isinstance(chain_ids, list): cs = f"_{''.join(chain_ids)}"
    name = Path(file_path).stem + cs
    cp = os.path.join(cache_dir, f"{name}.npz"); dp = os.path.join(cache_dir, f"{name}_dyn.npz")
    if os.path.exists(cp) and (not compute_dynamics or os.path.exists(dp)): return cp
    bb = parse_structure(file_path, chain_ids=chain_ids, min_len=min_len)
    if bb is None: return None
    L = bb['n_res']
    if L < min_len or L > max_len: return None
    cc = build_cochain_complex(bb, edge_cutoff=edge_cutoff)
    if compute_dynamics:
        dyn = compute_anm_dynamics(bb, n_modes=n_modes, n_conformers=n_conformers)
        if dyn is not None:
            mbb = {**bb, 'CA': dyn['medoid_coords']}
            cc = build_cochain_complex(mbb, edge_cutoff=edge_cutoff)
            pv = compute_pairwise_dist_var_vectorized(bb, dyn, cc['edge_src'], cc['edge_dst'], n_conformers=n_conformers)
            save_features({'msf': dyn['msf'], 'msf_empirical': dyn['msf_empirical'], 'pair_var': pv,
                'medoid_idx': np.array(dyn['medoid_idx']), 'eigenvalues': dyn['eigenvalues']}, dp)
    save_features(cc, cp)
    return cp
def process_dataset(data_dir, cache_dir, n_workers=4, per_chain=True, **kwargs):
    """Process all PDB/CIF files in data_dir to cache_dir. Returns list of cache paths."""
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, '*.pdb')) + glob.glob(os.path.join(data_dir, '*.cif'))
                   + glob.glob(os.path.join(data_dir, '*.pdb.gz')))
    logger.info(f"Processing {len(files)} structures with {n_workers} workers, dynamics={kwargs.get('compute_dynamics', False)}")
    paths = []
    if n_workers <= 1:
        for f in files:
            if per_chain:
                for ch in list_chains(f):
                    r = process_single_structure(f, cache_dir, chain_ids=ch, **kwargs)
                    if r: paths.append(r)
            else:
                r = process_single_structure(f, cache_dir, **kwargs)
                if r: paths.append(r)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for f in files:
                if per_chain:
                    for ch in list_chains(f):
                        fut = pool.submit(process_single_structure, f, cache_dir, chain_ids=ch, **kwargs)
                        futures[fut] = f"{f}:{ch}"
                else:
                    fut = pool.submit(process_single_structure, f, cache_dir, **kwargs)
                    futures[fut] = f
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r: paths.append(r)
                except Exception as e: logger.warning(f"Failed {futures[fut]}: {e}")
    logger.info(f"Processed {len(paths)} chains")
    return sorted(paths)
# ══════════════════════════════════════════════════════════════════════════════
# LMDB backend — zero-copy memory-mapped streaming (FIX: replaces .npz I/O)
# ══════════════════════════════════════════════════════════════════════════════
def build_lmdb(cache_paths, lmdb_path, map_size=100 * 1024**3):
    """Convert .npz cache files into a single LMDB for fast streaming.
    Each entry: key=index(bytes), value=pickle(dict of np arrays).
    Stores __lengths__ metadata for fast pre-filtering without full deserialization."""
    import lmdb
    os.makedirs(os.path.dirname(lmdb_path) or '.', exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, lock=False)
    valid_paths = []; lengths = []
    with env.begin(write=True) as txn:
        idx = 0
        for cp in cache_paths:
            data = load_features(cp)
            # merge dynamics if available
            dp = cp.replace('.npz', '_dyn.npz')
            if os.path.exists(dp):
                dy = load_features(dp)
                data['msf'] = dy['msf']
                data['pair_var'] = dy.get('pair_var', np.zeros(int(data['n_edges'])))
                data['has_dynamics'] = np.array(1, dtype=np.int64)
            else:
                data['msf'] = np.zeros(int(data['n_res']), dtype=np.float32)
                data['pair_var'] = np.zeros(int(data['n_edges']), dtype=np.float32)
                data['has_dynamics'] = np.array(0, dtype=np.int64)
            buf = pickle.dumps({k: v for k, v in data.items()}, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(str(idx).encode(), buf)
            lengths.append(int(data['n_res']))
            valid_paths.append(cp); idx += 1
        txn.put(b'__len__', str(idx).encode())
        # store lengths array for fast pre-filtering (avoids deserializing all entries)
        txn.put(b'__lengths__', pickle.dumps(lengths, protocol=pickle.HIGHEST_PROTOCOL))
    env.close()
    logger.info(f"Built LMDB: {lmdb_path} with {idx} entries")
    return lmdb_path
# ══════════════════════════════════════════════════════════════════════════════
# Datasets: LMDB (primary) + .npz fallback
# FIX #15: no infinite recursion on oversized proteins
# ══════════════════════════════════════════════════════════════════════════════
class LMDBDataset:
    """Zero-copy memory-mapped dataset from LMDB. OS handles page caching."""
    def __init__(self, lmdb_path, max_len=500):
        import lmdb
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=True, meminit=False)
        with self.env.begin() as txn: self._len = int(txn.get(b'__len__').decode())
        self.max_len = max_len
        # fast pre-filter using stored lengths (no full deserialization needed)
        self._valid_idx = []
        with self.env.begin() as txn:
            lengths_buf = txn.get(b'__lengths__')
            if lengths_buf is not None:
                # fast path: lengths stored during build_lmdb
                lengths = pickle.loads(lengths_buf)
                self._valid_idx = [i for i, l in enumerate(lengths) if l <= max_len]
            else:
                # fallback for old LMDB files: scan entries (slow)
                for i in range(self._len):
                    buf = txn.get(str(i).encode())
                    if buf is None: continue
                    data = pickle.loads(buf)
                    if int(data['n_res']) <= max_len: self._valid_idx.append(i)
        logger.info(f"LMDBDataset: {len(self._valid_idx)}/{self._len} proteins <= {max_len} residues")
    def __len__(self): return len(self._valid_idx)
    def __getitem__(self, idx):
        import torch
        real_idx = self._valid_idx[idx]
        with self.env.begin() as txn: buf = txn.get(str(real_idx).encode())
        data = pickle.loads(buf)
        r = {}
        for k in ['node_feat','edge_feat','bend_feat','torsion_feat','seq_idx','biochem_targets','torsion_biochem_targets',
                   'edge_src','edge_dst','bends','torsions',
                   'nbr0_src','nbr0_dst','nbr1_src','nbr1_dst','nbr2_src','nbr2_dst','nbr3_src','nbr3_dst',
                   'inc_01_edge','inc_01_node','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']:
            if k in data: r[k] = torch.from_numpy(data[k])
        r['n_res'] = torch.tensor(int(data['n_res']), dtype=torch.long)
        r['n_edges'] = torch.tensor(int(data['n_edges']), dtype=torch.long)
        r['n_bends'] = torch.tensor(int(data['n_bends']), dtype=torch.long)
        r['n_torsions'] = torch.tensor(int(data['n_torsions']), dtype=torch.long)
        r['msf'] = torch.from_numpy(data['msf'].astype(np.float32))
        r['pair_var'] = torch.from_numpy(data['pair_var'].astype(np.float32))
        r['has_dynamics'] = torch.tensor(int(data['has_dynamics']), dtype=torch.long)
        return r
class NPZDataset:
    """Fallback .npz dataset for when LMDB is not built yet.
    FIX #15: pre-filters by length at init — no recursion."""
    def __init__(self, cache_paths, max_len=500):
        self.max_len = max_len
        # pre-filter: only keep paths where n_res <= max_len
        self.paths = []
        for p in cache_paths:
            try:
                with np.load(p, allow_pickle=False) as f:
                    if int(f['n_res']) <= max_len: self.paths.append(p)
            except Exception: continue
        logger.info(f"NPZDataset: {len(self.paths)}/{len(cache_paths)} proteins <= {max_len} residues")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        import torch
        data = load_features(self.paths[idx]); r = {}
        for k in ['node_feat','edge_feat','bend_feat','torsion_feat','seq_idx','biochem_targets','torsion_biochem_targets',
                   'edge_src','edge_dst','bends','torsions',
                   'nbr0_src','nbr0_dst','nbr1_src','nbr1_dst','nbr2_src','nbr2_dst','nbr3_src','nbr3_dst',
                   'inc_01_edge','inc_01_node','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']:
            if k in data: r[k] = torch.from_numpy(data[k])
        r['n_res'] = torch.tensor(int(data['n_res']), dtype=torch.long)
        r['n_edges'] = torch.tensor(int(data['n_edges']), dtype=torch.long)
        r['n_bends'] = torch.tensor(int(data['n_bends']), dtype=torch.long)
        r['n_torsions'] = torch.tensor(int(data['n_torsions']), dtype=torch.long)
        dp = self.paths[idx].replace('.npz', '_dyn.npz')
        if os.path.exists(dp):
            dy = load_features(dp)
            r['msf'] = torch.from_numpy(dy['msf'].astype(np.float32))
            r['pair_var'] = torch.from_numpy(dy['pair_var'].astype(np.float32)) if 'pair_var' in dy else torch.zeros(int(data['n_edges']))
            r['has_dynamics'] = torch.tensor(1, dtype=torch.long)
        else:
            r['msf'] = torch.zeros(int(data['n_res'])); r['pair_var'] = torch.zeros(int(data['n_edges']))
            r['has_dynamics'] = torch.tensor(0, dtype=torch.long)
        return r
# ══════════════════════════════════════════════════════════════════════════════
# Collate (FIX #13: vectorized offset computation)
# ══════════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    import torch
    r = {}; bn = [b['n_res'].item() for b in batch]; be = [b['n_edges'].item() for b in batch]
    bb = [b['n_bends'].item() for b in batch]; bt = [b['n_torsions'].item() for b in batch]
    # precompute cumulative offsets
    off_n = np.concatenate([[0], np.cumsum(bn[:-1])]).astype(np.int64)
    off_e = np.concatenate([[0], np.cumsum(be[:-1])]).astype(np.int64)
    off_b = np.concatenate([[0], np.cumsum(bb[:-1])]).astype(np.int64)
    off_t = np.concatenate([[0], np.cumsum(bt[:-1])]).astype(np.int64)
    # plain cat fields (no offset needed)
    for k in ['node_feat','seq_idx','msf']: r[k] = torch.cat([b[k] for b in batch])
    for k in ['edge_feat','pair_var']: r[k] = torch.cat([b[k] for b in batch])
    for k in ['bend_feat','biochem_targets']: r[k] = torch.cat([b[k] for b in batch])
    r['torsion_feat'] = torch.cat([b['torsion_feat'] for b in batch])
    r['torsion_biochem_targets'] = torch.cat([b.get('torsion_biochem_targets', torch.zeros(b['n_torsions'].item(), 4)) for b in batch])
    # offset index fields: node, edge, bend, torsion offsets
    ok_n = {'edge_src','edge_dst','nbr0_src','nbr0_dst','inc_01_node'}
    ok_e = {'nbr1_src','nbr1_dst','inc_01_edge','inc_12_edge'}
    ok_b = {'nbr2_src','nbr2_dst','inc_12_bend','inc_23_bend'}
    ok_t = {'nbr3_src','nbr3_dst','inc_23_torsion'}
    for keys, offs in [(ok_n, off_n), (ok_e, off_e), (ok_b, off_b), (ok_t, off_t)]:
        for k in keys:
            r[k] = torch.cat([b[k] + int(offs[i]) for i, b in enumerate(batch)])
    # bends and torsions with node offset
    r['bends'] = torch.cat([b['bends'] + int(off_n[i]) for i, b in enumerate(batch)])
    r['torsions'] = torch.cat([b['torsions'] + int(off_n[i]) for i, b in enumerate(batch)])
    r['n_res'] = torch.tensor(bn); r['n_edges'] = torch.tensor(be)
    r['n_bends'] = torch.tensor(bb); r['n_torsions'] = torch.tensor(bt)
    r['has_dynamics'] = torch.stack([b['has_dynamics'] for b in batch])
    r['node_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(bn)])
    r['edge_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(be)])
    r['bend_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(bb)])
    r['torsion_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(bt)])
    return r
def get_dataloader(source, batch_size=1, shuffle=True, num_workers=2, max_len=500, pin_memory=True):
    """Create DataLoader from LMDB path (str ending .lmdb) or list of .npz paths."""
    from torch.utils.data import DataLoader
    if isinstance(source, str) and source.endswith('.lmdb'):
        ds = LMDBDataset(source, max_len)
    elif isinstance(source, list):
        ds = NPZDataset(source, max_len)
    else:
        raise ValueError(f"source must be .lmdb path or list of .npz paths, got {type(source)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=pin_memory, drop_last=False,
        persistent_workers=num_workers > 0)
# ══════════════════════════════════════════════════════════════════════════════
# Train/Val/Test splits
# ══════════════════════════════════════════════════════════════════════════════
def create_splits(cache_paths, output_dir, val_frac=0.1, test_frac=0.1, seed=42, split_file=None):
    if split_file and os.path.exists(split_file):
        with open(split_file) as f: return json.load(f)
    np.random.seed(seed); paths = list(cache_paths); np.random.shuffle(paths)
    n = len(paths); nv = int(n * val_frac); nt = int(n * test_frac)
    splits = {'test': paths[:nt], 'val': paths[nt:nt+nv], 'train': paths[nt+nv:]}
    sf = os.path.join(output_dir, 'splits.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(sf, 'w') as f: json.dump(splits, f)
    return splits
