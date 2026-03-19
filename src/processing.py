"""META processing: PDB/CIF parsing, cochain complex construction (ranks 0-3),
adjacency/coadjacency/incidence, ProDy dynamics, dataset splits.
Cochains: 0=residues, 1=edges, 2=bends(angle), 3=torsions(dihedral)."""
import numpy as np, os, glob, json, logging, warnings
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
# PDB/CIF Parser (unchanged from v2)
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
    Nm, CAm, Cm, rnm = {}, {}, {}, {}
    seen = set()
    for a in recs:
        if chain_ids is not None and a['chain'] not in chain_ids: continue
        key = (a['chain'], a['resi'], a['icode'])
        if (key, a['atom']) in seen: continue
        seen.add((key, a['atom']))
        co = np.array([a['x'], a['y'], a['z']])
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
# ══════════════════════════════════════════════════════════════════════════════
def build_cochain_complex(backbone: Dict, edge_cutoff: float = 8.0, max_chain_enc: int = 16) -> Dict:
    """Build cochain complex from backbone.
    Rank 0: residues (d0=23 single-chain, +16 chain enc for multi-chain)
    Rank 1: edges/contacts (d1=38)
    Rank 2: bends (i,j,k) with j central, edges i-j and j-k exist, i<k (d2=1, cos angle)
    Rank 3: torsions (i,j,k,l) from two bends sharing edge j-k (d3=2, sin/cos dihedral)"""
    import torch as _t
    N, CA, C, seq = backbone['N'], backbone['CA'], backbone['C'], backbone['seq']
    L = len(CA)
    ci = backbone.get('chain_idx', np.zeros(L, dtype=np.int64))
    rn = backbone.get('res_numbers', np.arange(L, dtype=np.int64))
    nc = int(ci.max()) + 1
    seq_idx = aa_to_idx(seq)
    # ── Rank 0: node features (NO DSSP) ──
    phi, psi, omega = compute_dihedrals(N, CA, C, chain_idx=ci)
    phi[np.isnan(phi)] = 0; psi[np.isnan(psi)] = 0; omega[np.isnan(omega)] = 0
    dih = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi), np.sin(omega), np.cos(omega)], -1)  # (L,6)
    ba_rbf = rbf_encode(_t.tensor(compute_bond_angle(N, CA, C), dtype=_t.float32), 1.5, 2.5, 16).numpy()  # (L,16)
    sasa = compute_sasa_shrake_rupley(CA)[:, None]  # (L,1)
    nf_parts = [dih, ba_rbf, sasa]  # 6+16+1 = 23 (no explicit chain encoding: implicit via covalent bonds)
    node_feat = np.concatenate(nf_parts, -1).astype(np.float32)
    d_node = node_feat.shape[1]
    # ── Rank 1: edge construction (KD-tree) ──
    CB = compute_virtual_cbeta(N, CA, C)
    esrc, edst = compute_edges_kdtree(CB, edge_cutoff) if L > 50 else _dense_edges(CB, edge_cutoff)
    n_edges = len(esrc)
    if n_edges == 0:
        esrc, edst = compute_edges_kdtree(CA, edge_cutoff + 2.0) if L > 50 else _dense_edges(CA, edge_cutoff + 2.0)
        n_edges = len(esrc)
    # edge features: dist_rbf(16) + local_dir(3) + seq_sep(16) + covalent(2) = 37
    cb_d = np.linalg.norm(CB[edst] - CB[esrc], axis=-1)
    dist_rbf = rbf_encode(_t.tensor(cb_d, dtype=_t.float32), 0, 20, 16).numpy()
    dir_raw = CA[edst] - CA[esrc]  # FROM src TO dst: convention
    dir_n = dir_raw / (np.linalg.norm(dir_raw, axis=-1, keepdims=True) + 1e-8)
    frames = compute_local_frame(N, CA, C)
    loc_dir = project_to_local_frame(dir_n, frames, esrc)
    sep = compute_seq_separation(esrc, edst, ci, rn)
    sep_enc = sinusoidal_encode(_t.tensor(sep, dtype=_t.float32), 16).numpy()
    cov_oh = compute_covalent_onehot(esrc, edst, ci, rn)  # implicitly encodes chain boundaries
    edge_feat = np.concatenate([dist_rbf, loc_dir, sep_enc, cov_oh], -1).astype(np.float32)
    d_edge = edge_feat.shape[1]
    # edge lookup: (src_node, dst_node) -> edge_index
    e2i = {}
    for ei in range(n_edges): e2i[(int(esrc[ei]), int(edst[ei]))] = ei
    # build undirected adjacency for bend construction
    adj = defaultdict(set)
    for i, j in zip(esrc.tolist(), edst.tolist()): adj[i].add(j)
    # ── Rank 2: bends (i,j,k) with j central, i<k ──
    bends = []
    for j in range(L):
        nbrs = sorted(adj[j])
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                i, k = nbrs[ai], nbrs[bi]
                bends.append((i, j, k))
    bends = np.array(bends, dtype=np.int64) if bends else np.zeros((0, 3), dtype=np.int64)
    n_bends = len(bends)
    # bend features: cos(angle at j)
    if n_bends > 0:
        bend_cos = compute_bend_cosine(CA, bends[:, 0], bends[:, 1], bends[:, 2])
        bend_feat = bend_cos[:, None]  # (B, 1)
    else:
        bend_feat = np.zeros((0, 1), dtype=np.float32)
    # bend -> edge mapping for incidence
    bend_to_edges = []  # list of (bend_idx, edge_idx)
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        for e in [(i, j), (j, i), (j, k), (k, j)]:
            if e in e2i: bend_to_edges.append((bi, e2i[e]))
    # ── Rank 3: torsions (i,j,k,l) from two bends sharing edge j-k ──
    # index bends by their shared edge (central-node, other-node pairs)
    # For bend (i,j,k): the two edges are (i,j) and (j,k)
    # Two bends share edge (j,k) if: bend1 = (i,j,k) and bend2 = (m,k,n) where m=j
    # More precisely: bend1 has edge (j,k), bend2 has edge (j,k) too
    # So we index: for each directed edge (a,b), which bends contain it?
    edge_key_to_bends = defaultdict(list)  # (min(a,b), max(a,b)) -> list of bend indices
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        # edges of this bend are (i,j) and (j,k)
        edge_key_to_bends[(min(i,j), max(i,j))].append(bi)
        edge_key_to_bends[(min(j,k), max(j,k))].append(bi)
    torsions = []  # (i,j,k,l) + (bend1_idx, bend2_idx)
    torsion_bend_pairs = []
    for ekey, bindices in edge_key_to_bends.items():
        if len(bindices) < 2: continue
        for ai in range(len(bindices)):
            for bi_idx in range(ai + 1, len(bindices)):
                b1, b2 = bindices[ai], bindices[bi_idx]
                # get the 4 nodes: union of both bends
                nodes1 = set(bends[b1].tolist()); nodes2 = set(bends[b2].tolist())
                shared = nodes1 & nodes2
                if len(shared) != 2: continue  # must share exactly the edge (2 nodes)
                all_nodes = nodes1 | nodes2
                if len(all_nodes) != 4: continue  # must have 4 distinct nodes
                # order: i,j,k,l where (j,k) is shared edge
                j_node, k_node = sorted(shared)
                outer1 = (nodes1 - shared).pop()
                outer2 = (nodes2 - shared).pop()
                torsions.append((outer1, j_node, k_node, outer2))
                torsion_bend_pairs.append((b1, b2))
    torsions = np.array(torsions, dtype=np.int64) if torsions else np.zeros((0, 4), dtype=np.int64)
    n_torsions = len(torsions)
    if n_torsions > 0:
        sin_d, cos_d = compute_dihedral_4point(CA, torsions[:, 0], torsions[:, 1], torsions[:, 2], torsions[:, 3])
        torsion_feat = np.stack([sin_d, cos_d], -1)  # (D, 2)
    else:
        torsion_feat = np.zeros((0, 2), dtype=np.float32)
    # ── Neighbourhood: adjacency + coadjacency per rank ──
    # Rank 0: adj = edge graph (already have esrc, edst)
    # coadj = both in same edge = same as adj for rank 0
    nbr0_src, nbr0_dst = esrc.copy(), edst.copy()
    # Rank 1: adj = share a node; coadj = both in same bend
    node2edges = defaultdict(list)
    for ei in range(n_edges): node2edges[int(esrc[ei])].append(ei); node2edges[int(edst[ei])].append(ei)
    eadj = set()  # edge adjacency
    for node, eidxs in node2edges.items():
        for a in eidxs:
            for b in eidxs:
                if a != b: eadj.add((a, b))
    ecoadj = set()  # edge coadjacency: both in same bend
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        bend_edges = set()
        for e in [(i,j),(j,i),(j,k),(k,j)]:
            if e in e2i: bend_edges.add(e2i[e])
        for a in bend_edges:
            for b in bend_edges:
                if a != b: ecoadj.add((a, b))
    enbr = sorted(eadj | ecoadj)
    nbr1_src = np.array([p[0] for p in enbr], dtype=np.int64) if enbr else np.zeros(0, dtype=np.int64)
    nbr1_dst = np.array([p[1] for p in enbr], dtype=np.int64) if enbr else np.zeros(0, dtype=np.int64)
    # Rank 2: adj = share an edge; coadj = both in same torsion
    bend_edge_sets = []  # for each bend, set of edge keys
    for bi in range(n_bends):
        i, j, k = int(bends[bi, 0]), int(bends[bi, 1]), int(bends[bi, 2])
        es = set()
        for e in [(i,j),(j,i),(j,k),(k,j)]:
            if e in e2i: es.add((min(e), max(e)))
        bend_edge_sets.append(es)
    badj = set()
    ekey_to_bends_list = defaultdict(list)
    for bi in range(n_bends):
        for ek in bend_edge_sets[bi]: ekey_to_bends_list[ek].append(bi)
    for ek, bis in ekey_to_bends_list.items():
        for a in bis:
            for b in bis:
                if a != b: badj.add((a, b))
    bcoadj = set()  # both in same torsion
    for ti in range(n_torsions):
        b1, b2 = torsion_bend_pairs[ti]
        bcoadj.add((b1, b2)); bcoadj.add((b2, b1))
    bnbr = sorted(badj | bcoadj)
    nbr2_src = np.array([p[0] for p in bnbr], dtype=np.int64) if bnbr else np.zeros(0, dtype=np.int64)
    nbr2_dst = np.array([p[1] for p in bnbr], dtype=np.int64) if bnbr else np.zeros(0, dtype=np.int64)
    # Rank 3: adj = share a bend; no coadjacency (no rank 4)
    tbp_arr = np.array(torsion_bend_pairs, dtype=np.int64) if torsion_bend_pairs else np.zeros((0, 2), dtype=np.int64)
    bend_to_torsions = defaultdict(list)
    for ti in range(n_torsions):
        for bi in tbp_arr[ti]: bend_to_torsions[int(bi)].append(ti)
    tadj = set()
    for bi, tis in bend_to_torsions.items():
        for a in tis:
            for b in tis:
                if a != b: tadj.add((a, b))
    tnbr = sorted(tadj)
    nbr3_src = np.array([p[0] for p in tnbr], dtype=np.int64) if tnbr else np.zeros(0, dtype=np.int64)
    nbr3_dst = np.array([p[1] for p in tnbr], dtype=np.int64) if tnbr else np.zeros(0, dtype=np.int64)
    # ── Incidence matrices (sparse) ──
    # I_01: node -> edge
    inc_01_edge = np.concatenate([np.arange(n_edges), np.arange(n_edges)]).astype(np.int64)
    inc_01_node = np.concatenate([esrc, edst]).astype(np.int64)
    # I_12: edge -> bend
    if bend_to_edges:
        bte = np.array(bend_to_edges, dtype=np.int64)
        bte = np.unique(bte, axis=0)
        inc_12_bend, inc_12_edge = bte[:, 0], bte[:, 1]
    else:
        inc_12_bend = inc_12_edge = np.zeros(0, dtype=np.int64)
    # I_23: bend -> torsion
    if n_torsions > 0:
        i23 = []
        for ti in range(n_torsions):
            b1, b2 = torsion_bend_pairs[ti]
            i23.append((ti, b1)); i23.append((ti, b2))
        i23 = np.unique(np.array(i23, dtype=np.int64), axis=0)
        inc_23_torsion, inc_23_bend = i23[:, 0], i23[:, 1]
    else:
        inc_23_torsion = inc_23_bend = np.zeros(0, dtype=np.int64)
    # biochem targets (3 residues per bend, 4 residues per torsion)
    biochem_targets = compute_triangle_properties(seq_idx, bends) if n_bends > 0 else np.zeros((0, 4), dtype=np.float32)
    torsion_biochem_targets = compute_torsion_properties(seq_idx, torsions) if n_torsions > 0 else np.zeros((0, 4), dtype=np.float32)
    return {
        'n_res': L, 'n_edges': n_edges, 'n_bends': n_bends, 'n_torsions': n_torsions,
        'd_node': d_node, 'd_edge': d_edge,
        'seq_idx': seq_idx, 'seq': np.array(list(seq)),
        'node_feat': node_feat, 'edge_feat': edge_feat, 'bend_feat': bend_feat, 'torsion_feat': torsion_feat,
        'edge_src': esrc, 'edge_dst': edst, 'bends': bends, 'torsions': torsions,
        'chain_idx': ci, 'CA': CA.astype(np.float32),
        # neighbourhood (adj ∪ coadj)
        'nbr0_src': nbr0_src, 'nbr0_dst': nbr0_dst,
        'nbr1_src': nbr1_src, 'nbr1_dst': nbr1_dst,
        'nbr2_src': nbr2_src, 'nbr2_dst': nbr2_dst,
        'nbr3_src': nbr3_src, 'nbr3_dst': nbr3_dst,
        # incidence
        'inc_01_edge': inc_01_edge, 'inc_01_node': inc_01_node,
        'inc_12_bend': inc_12_bend, 'inc_12_edge': inc_12_edge,
        'inc_23_torsion': inc_23_torsion, 'inc_23_bend': inc_23_bend,
        'biochem_targets': biochem_targets,
        'torsion_biochem_targets': torsion_biochem_targets,
    }
# backward compat alias
build_combinatorial_complex = build_cochain_complex
def _dense_edges(coords, cutoff):
    L = len(coords)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    m = (d < cutoff) & (~np.eye(L, dtype=bool))
    s, t = np.where(m)
    return s.astype(np.int64), t.astype(np.int64)
# ══════════════════════════════════════════════════════════════════════════════
# ProDy ANM Dynamics (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def compute_anm_dynamics(backbone, n_modes=20, n_conformers=1000, cutoff=15.0, temperature=300.0):
    try:
        import prody; prody.confProDy(verbosity='none')
    except ImportError: return None
    CA = backbone['CA']; L = len(CA)
    anm = prody.ANM('protein'); anm.buildHessian(CA, cutoff=cutoff)
    try: anm.calcModes(n_modes=min(n_modes, 3*L-7))
    except Exception: return None
    ev = anm.getEigvals(); evc = anm.getEigvecs(); K = len(ev)
    kBT = 1.38064852e-23 * temperature; g = 1.0
    msf = np.zeros(L, dtype=np.float64)
    for m in range(K): msf += np.sum(evc[:, m].reshape(L, 3)**2, -1) / ev[m]
    msf = (msf * kBT / g).astype(np.float32)
    np.random.seed(42)
    conf = np.zeros((n_conformers, L, 3), dtype=np.float32)
    for s in range(n_conformers):
        disp = sum(np.random.normal(0, np.sqrt(kBT/(g*ev[m]))) * evc[:, m].reshape(L, 3) for m in range(K))
        conf[s] = (CA + disp).astype(np.float32)
    sub_n = min(200, n_conformers); si = np.random.choice(n_conformers, sub_n, replace=False)
    flat = (conf[si] - conf[si].mean(1, keepdims=True)).reshape(sub_n, -1)
    ad = np.zeros(sub_n)
    for s in range(0, sub_n, 50):
        e = min(s+50, sub_n)
        ad[s:e] = np.sqrt(np.mean((flat[s:e, None] - flat[None])**2, -1)).mean(-1)
    mi = si[np.argmin(ad)]
    return {'msf': msf, 'msf_empirical': np.mean(np.sum((conf - conf.mean(0)[None])**2, -1), 0).astype(np.float32),
        'medoid_coords': conf[mi], 'medoid_idx': int(mi), 'eigenvalues': ev.astype(np.float32), 'n_modes': K}
def compute_pairwise_dist_var(backbone, dynamics, esrc, edst, n_conformers=500, n_modes=20, cutoff=15.0, temperature=300.0):
    try: import prody; prody.confProDy(verbosity='none')
    except ImportError: return np.zeros(len(esrc), dtype=np.float32)
    CA = backbone['CA']; L = len(CA)
    anm = prody.ANM('protein'); anm.buildHessian(CA, cutoff=cutoff)
    anm.calcModes(n_modes=min(n_modes, 3*L-7))
    evc = anm.getEigvecs(); ev = anm.getEigvals(); K = len(ev)
    kBT = 1.38064852e-23*temperature; g = 1.0; np.random.seed(42); ns = min(n_conformers, 500)
    dists = np.zeros((ns, len(esrc)), dtype=np.float32)
    for s in range(ns):
        disp = sum(np.random.normal(0, np.sqrt(kBT/(g*ev[m]))) * evc[:, m].reshape(L, 3) for m in range(K))
        dists[s] = np.linalg.norm((CA+disp)[edst] - (CA+disp)[esrc], -1)
    return np.var(dists, 0).astype(np.float32)
# ══════════════════════════════════════════════════════════════════════════════
# Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def process_single_structure(file_path, cache_dir, chain_ids=None, edge_cutoff=8.0,
                             compute_dynamics=True, n_modes=20, n_conformers=1000, min_len=30, max_len=500):
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
            pv = compute_pairwise_dist_var(bb, dyn, cc['edge_src'], cc['edge_dst'], n_conformers=n_conformers, n_modes=n_modes)
            save_features({'msf': dyn['msf'], 'msf_empirical': dyn['msf_empirical'], 'pair_var': pv,
                'medoid_idx': np.array(dyn['medoid_idx']), 'eigenvalues': dyn['eigenvalues']}, dp)
    save_features(cc, cp)
    return cp
process_single_pdb = process_single_structure
def process_pdb_directory(pdb_dir, cache_dir, compute_dynamics=True, n_workers=4, edge_cutoff=8.0,
                          n_modes=20, n_conformers=1000, per_chain=True, min_len=30, max_len=500):
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(sum([glob.glob(os.path.join(pdb_dir, e)) for e in ('*.pdb','*.ent','*.cif','*.mmcif')], []))
    logger.info(f"Found {len(files)} structure files")
    tasks = []
    if per_chain:
        for f in files:
            try:
                for ch in list_chains(f): tasks.append((f, ch))
            except Exception: tasks.append((f, None))
    else:
        tasks = [(f, None) for f in files]
    results = []
    if n_workers <= 1:
        for f, ch in tasks:
            r = process_single_structure(f, cache_dir, ch, edge_cutoff, compute_dynamics, n_modes, n_conformers, min_len, max_len)
            if r: results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_single_structure, f, cache_dir, ch, edge_cutoff, compute_dynamics, n_modes, n_conformers, min_len, max_len): (f, ch) for f, ch in tasks}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if r: results.append(r)
                except Exception as e: logger.error(f"Error: {futs[fut]}: {e}")
    logger.info(f"Processed {len(results)}/{len(tasks)}")
    return sorted(results)
def create_splits(cache_paths, output_dir, val_frac=0.1, test_frac=0.1, seed=42, split_file=None):
    sp = split_file or os.path.join(output_dir, 'splits.json')
    if os.path.exists(sp):
        with open(sp) as f: return json.load(f)
    np.random.seed(seed); n = len(cache_paths); idx = np.random.permutation(n)
    nt = max(1, int(n*test_frac)); nv = max(1, int(n*val_frac))
    splits = {'test': [cache_paths[i] for i in idx[:nt]], 'val': [cache_paths[i] for i in idx[nt:nt+nv]], 'train': [cache_paths[i] for i in idx[nt+nv:]]}
    os.makedirs(output_dir, exist_ok=True)
    with open(sp, 'w') as f: json.dump(splits, f, indent=2)
    return splits
# ══════════════════════════════════════════════════════════════════════════════
# Streaming Dataset
# ══════════════════════════════════════════════════════════════════════════════
class METADataset:
    def __init__(self, cache_paths, max_len=500):
        self.paths = cache_paths; self.max_len = max_len
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        import torch
        data = load_features(self.paths[idx])
        nr = int(data['n_res'])
        if nr > self.max_len: return self.__getitem__(np.random.randint(len(self)))
        r = {}
        for k in ['node_feat','edge_feat','bend_feat','torsion_feat','seq_idx','biochem_targets','torsion_biochem_targets',
                   'edge_src','edge_dst','bends','torsions',
                   'nbr0_src','nbr0_dst','nbr1_src','nbr1_dst','nbr2_src','nbr2_dst','nbr3_src','nbr3_dst',
                   'inc_01_edge','inc_01_node','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']:
            if k in data: r[k] = torch.from_numpy(data[k])
        r['n_res'] = torch.tensor(nr, dtype=torch.long)
        r['n_edges'] = torch.tensor(int(data['n_edges']), dtype=torch.long)
        r['n_bends'] = torch.tensor(int(data['n_bends']), dtype=torch.long)
        r['n_torsions'] = torch.tensor(int(data['n_torsions']), dtype=torch.long)
        r['d_node'] = torch.tensor(int(data.get('d_node', data['node_feat'].shape[1])), dtype=torch.long)
        r['d_edge'] = torch.tensor(int(data.get('d_edge', data['edge_feat'].shape[1])), dtype=torch.long)
        dp = self.paths[idx].replace('.npz', '_dyn.npz')
        if os.path.exists(dp):
            dy = load_features(dp)
            r['msf'] = torch.from_numpy(dy['msf'])
            r['pair_var'] = torch.from_numpy(dy['pair_var']) if 'pair_var' in dy else torch.zeros(int(data['n_edges']))
            r['has_dynamics'] = torch.tensor(1, dtype=torch.long)
        else:
            r['msf'] = torch.zeros(nr); r['pair_var'] = torch.zeros(int(data['n_edges'])); r['has_dynamics'] = torch.tensor(0, dtype=torch.long)
        return r
def collate_fn(batch):
    import torch
    r = {}; bn = [b['n_res'].item() for b in batch]; be = [b['n_edges'].item() for b in batch]
    bb = [b['n_bends'].item() for b in batch]; bt = [b['n_torsions'].item() for b in batch]
    for k in ['node_feat','seq_idx','msf']: r[k] = torch.cat([b[k] for b in batch])
    for k in ['edge_feat','pair_var']: r[k] = torch.cat([b[k] for b in batch])
    for k in ['bend_feat','biochem_targets']: r[k] = torch.cat([b[k] for b in batch])
    r['torsion_feat'] = torch.cat([b['torsion_feat'] for b in batch])
    r['torsion_biochem_targets'] = torch.cat([b.get('torsion_biochem_targets', torch.zeros(b['n_torsions'].item(), 4)) for b in batch])
    # offset indices
    ok_n = ['edge_src','edge_dst','nbr0_src','nbr0_dst','inc_01_node']
    ok_e = ['nbr1_src','nbr1_dst','inc_01_edge','inc_12_edge']
    ok_b = ['nbr2_src','nbr2_dst','inc_12_bend','inc_23_bend']
    ok_t = ['nbr3_src','nbr3_dst','inc_23_torsion']
    for k in ok_n + ok_e + ok_b + ok_t:
        parts = []; no = eo = bo = to = 0
        for i, b in enumerate(batch):
            v = b[k]
            if k in ok_n: v = v + no
            elif k in ok_e: v = v + eo
            elif k in ok_b: v = v + bo
            elif k in ok_t: v = v + to
            parts.append(v); no += bn[i]; eo += be[i]; bo += bb[i]; to += bt[i]
        r[k] = torch.cat(parts)
    # bends and torsions with node offset
    bp = []; tp = []; no = 0
    for i, b in enumerate(batch): bp.append(b['bends']+no); tp.append(b['torsions']+no); no += bn[i]
    r['bends'] = torch.cat(bp); r['torsions'] = torch.cat(tp)
    r['n_res'] = torch.tensor(bn); r['n_edges'] = torch.tensor(be)
    r['n_bends'] = torch.tensor(bb); r['n_torsions'] = torch.tensor(bt)
    r['has_dynamics'] = torch.stack([b['has_dynamics'] for b in batch])
    r['node_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(bn)])
    r['edge_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(be)])
    r['bend_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(bb)])
    r['torsion_batch'] = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(bt)])
    return r
def get_dataloader(cache_paths, batch_size=1, shuffle=True, num_workers=2, max_len=500, pin_memory=True):
    from torch.utils.data import DataLoader
    return DataLoader(METADataset(cache_paths, max_len), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=False, persistent_workers=num_workers > 0)
