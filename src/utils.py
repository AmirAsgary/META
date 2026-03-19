"""META utility functions: geometry, encodings, physicochemistry, cochain helpers.
Supports single-chain, multi-chain, chain breaks, non-standard residues."""
import numpy as np, os, hashlib, logging
from typing import Tuple, Optional, Dict, List
def _torch():
    import torch; return torch
logger = logging.getLogger(__name__)
# ── Amino acid mapping ──
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
NUM_AA = 20
THREE_TO_ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
    'MSE':'M','SEC':'C','CSO':'C','CSD':'C','CME':'C','OCS':'C','CSS':'C',
    'HSD':'H','HSE':'H','HSP':'H','HID':'H','HIE':'H','HIP':'H',
    'TPO':'T','SEP':'S','PTR':'Y','MLY':'K','M3L':'K','ALY':'K',
    'HYP':'P','PCA':'E','CGU':'E',
    'AIB':'A','DAL':'A','DPN':'F','DLE':'L','DVA':'V','UNK':'G','XAA':'G',
}
KD_HYDRO = dict(zip(AA_ORDER, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
CHARGE_PH74 = dict(zip(AA_ORDER, [0,0,-1,-1,0,0,0.1,0,1,0,0,0,0,0,1,0,0,0,0,0]))
VDW_VOLUME = dict(zip(AA_ORDER, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
HBOND_DA = dict(zip(AA_ORDER, [0,0,2,2,0,0,2,0,2,0,1,2,0,2,4,1,1,0,2,1]))
def aa_to_idx(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX.get(c, NUM_AA) for c in seq], dtype=np.int64)
def idx_to_aa(idx: np.ndarray) -> str:
    return ''.join(AA_ORDER[i] if 0 <= i < NUM_AA else 'X' for i in idx)
def three_to_one(resname: str) -> str:
    return THREE_TO_ONE.get(resname.strip(), 'X')
# ── Torch-based encodings (lazy import) ──
def rbf_encode(values, d_min: float = 0.0, d_max: float = 20.0, n_bins: int = 16):
    torch = _torch()
    mu = torch.linspace(d_min, d_max, n_bins, device=values.device, dtype=values.dtype)
    sigma = (d_max - d_min) / n_bins
    return torch.exp(-0.5 * ((values.unsqueeze(-1) - mu) / sigma) ** 2)
def sinusoidal_encode(values, n_dim: int = 16):
    torch = _torch()
    half = n_dim // 2
    freq = torch.exp(torch.arange(half, device=values.device, dtype=values.dtype) * -(np.log(10000.0) / half))
    angles = values.unsqueeze(-1) * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)
# ── Chain break detection ──
def detect_chain_breaks(CA: np.ndarray, chain_idx: np.ndarray, thresh: float = 4.5) -> np.ndarray:
    """Returns (L,) bool: True = starts a new segment."""
    L = len(CA)
    brk = np.zeros(L, dtype=bool); brk[0] = True
    if L < 2: return brk
    brk[1:] = (chain_idx[1:] != chain_idx[:-1]) | (np.linalg.norm(CA[1:] - CA[:-1], axis=-1) > thresh)
    return brk
# ── Backbone dihedrals (chain-boundary-aware) ──
def compute_dihedrals(N, CA, C, chain_idx=None, thresh=4.5):
    """Compute phi, psi, omega. NaN at chain boundaries."""
    def _dih(p0, p1, p2, p3):
        b0, b1, b2 = p1-p0, p2-p1, p3-p2
        n1 = np.cross(b0, b1); n2 = np.cross(b1, b2)
        n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
        n2 = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
        m1 = np.cross(n1, b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8))
        return np.arctan2(np.sum(m1*n2, -1), np.sum(n1*n2, -1))
    L = len(CA)
    phi, psi, omega = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
    if L < 2: return phi, psi, omega
    if chain_idx is None: chain_idx = np.zeros(L, dtype=np.int64)
    vp = ~detect_chain_breaks(CA, chain_idx, thresh)[1:]  # (L-1,) valid pairs
    phi[1:] = np.where(vp, _dih(C[:-1], N[1:], CA[1:], C[1:]), np.nan)
    psi[:-1] = np.where(vp, _dih(N[:-1], CA[:-1], C[:-1], N[1:]), np.nan)
    omega[:-1] = np.where(vp, _dih(CA[:-1], C[:-1], N[1:], CA[1:]), np.nan)
    return phi, psi, omega
# ── Bond angle, Cbeta, local frame ──
def compute_bond_angle(N, CA, C):
    v1 = N - CA; v2 = C - CA
    return np.arccos(np.clip(np.sum(v1*v2, -1) / (np.linalg.norm(v1, -1) * np.linalg.norm(v2, -1) + 1e-8), -1, 1))
def compute_virtual_cbeta(N, CA, C):
    n = N - CA; c = C - CA
    cr = np.cross(n, c); cr = cr / (np.linalg.norm(cr, axis=-1, keepdims=True) + 1e-8)
    d = np.cross(cr, n - c); d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
    return CA + (-0.58273431 * (n + c) + 1.20229 * d)
def compute_local_frame(N, CA, C):
    e1 = C - CA; e1 = e1 / (np.linalg.norm(e1, axis=-1, keepdims=True) + 1e-8)
    v = N - CA; e2 = v - np.sum(v*e1, axis=-1, keepdims=True) * e1
    e2 = e2 / (np.linalg.norm(e2, axis=-1, keepdims=True) + 1e-8)
    return np.stack([e1, e2, np.cross(e1, e2)], axis=-1)
def project_to_local_frame(direction, frames, idx):
    return np.einsum('ei,eij->ej', direction, frames[idx])
# ── Bend angle cosine (2-cochain feature) ──
def compute_bend_cosine(coords: np.ndarray, bend_i: np.ndarray, bend_j: np.ndarray, bend_k: np.ndarray) -> np.ndarray:
    """Cosine of angle at central node j in bend (i,j,k). coords: (L,3), indices: (B,). Returns (B,)."""
    v1 = coords[bend_i] - coords[bend_j]  # j->i
    v2 = coords[bend_k] - coords[bend_j]  # j->k
    cos_a = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, -1) * np.linalg.norm(v2, -1) + 1e-8)
    return np.clip(cos_a, -1.0, 1.0).astype(np.float32)
# ── Dihedral angle (3-cochain feature) ──
def compute_dihedral_4point(coords: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray,
                            idx_k: np.ndarray, idx_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dihedral angle between planes (i,j,k) and (j,k,l). Returns (sin, cos) each (D,)."""
    p0, p1, p2, p3 = coords[idx_i], coords[idx_j], coords[idx_k], coords[idx_l]
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b0, b1); n2 = np.cross(b1, b2)
    n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
    m1 = np.cross(n1, b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8))
    cos_d = np.clip(np.sum(n1 * n2, -1), -1, 1)
    sin_d = np.clip(np.sum(m1 * n2, -1), -1, 1)
    return sin_d.astype(np.float32), cos_d.astype(np.float32)
# ── Covalent bond encoding ──
def compute_covalent_onehot(edge_src: np.ndarray, edge_dst: np.ndarray,
                            chain_idx: np.ndarray, res_numbers: np.ndarray) -> np.ndarray:
    """2-dim one-hot: [1,0]=covalent (sequential in same chain), [0,1]=non-covalent. Returns (E,2)."""
    same_chain = chain_idx[edge_src] == chain_idx[edge_dst]
    sequential = np.abs(res_numbers[edge_src].astype(np.int64) - res_numbers[edge_dst].astype(np.int64)) == 1
    is_covalent = same_chain & sequential
    onehot = np.zeros((len(edge_src), 2), dtype=np.float32)
    onehot[is_covalent, 0] = 1.0; onehot[~is_covalent, 1] = 1.0
    return onehot
# ── Chain-aware sequence separation ──
def compute_seq_separation(edge_src, edge_dst, chain_idx, res_numbers=None, inter_val=999.0):
    same = chain_idx[edge_src] == chain_idx[edge_dst]
    sep = np.abs(edge_dst.astype(np.float32) - edge_src.astype(np.float32)) if res_numbers is None \
        else np.abs(res_numbers[edge_dst].astype(np.float32) - res_numbers[edge_src].astype(np.float32))
    return np.where(same, sep, inter_val)
# ── KD-tree edges ──
def compute_edges_kdtree(coords, cutoff):
    from scipy.spatial import cKDTree
    pairs = cKDTree(coords).query_pairs(cutoff, output_type='ndarray')
    if len(pairs) == 0: return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64), np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
# ── Triangle properties (for biochemistry targets, uses 3 residues of bend) ──
def compute_triangle_properties(seq_idx, triangles):
    if len(triangles) == 0: return np.zeros((0, 4), dtype=np.float32)
    safe = np.clip(seq_idx, 0, NUM_AA - 1)
    aa = [AA_ORDER[i] for i in safe]
    h = np.array([KD_HYDRO[a] for a in aa]); c = np.array([CHARGE_PH74[a] for a in aa])
    v = np.array([VDW_VOLUME[a] for a in aa]); b = np.array([HBOND_DA[a] for a in aa])
    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return np.stack([(h[i]+h[j]+h[k])/3, c[i]+c[j]+c[k], (v[i]+v[j]+v[k])/3, b[i]+b[j]+b[k]], -1).astype(np.float32)
# ── Torsion properties (for biochemistry targets, uses 4 residues of torsion) ──
def compute_torsion_properties(seq_idx, torsions):
    """Physicochemical properties from 4 constituent residues of each torsion."""
    if len(torsions) == 0: return np.zeros((0, 4), dtype=np.float32)
    safe = np.clip(seq_idx, 0, NUM_AA - 1)
    aa = [AA_ORDER[i] for i in safe]
    h = np.array([KD_HYDRO[a] for a in aa]); c = np.array([CHARGE_PH74[a] for a in aa])
    v = np.array([VDW_VOLUME[a] for a in aa]); b = np.array([HBOND_DA[a] for a in aa])
    i, j, k, l = torsions[:, 0], torsions[:, 1], torsions[:, 2], torsions[:, 3]
    return np.stack([(h[i]+h[j]+h[k]+h[l])/4, c[i]+c[j]+c[k]+c[l],
                     (v[i]+v[j]+v[k]+v[l])/4, b[i]+b[j]+b[k]+b[l]], -1).astype(np.float32)
# ── SASA (KD-tree accelerated) ──
def compute_sasa_shrake_rupley(coords, radii=None, n_points=100, probe=1.4):
    L = len(coords)
    if radii is None: radii = np.full(L, 1.8)
    r = radii + probe
    idx = np.arange(n_points, dtype=float) + 0.5
    phi_p = np.arccos(1 - 2*idx/n_points); theta_p = np.pi*(1+5**0.5)*idx
    sphere = np.stack([np.sin(phi_p)*np.cos(theta_p), np.sin(phi_p)*np.sin(theta_p), np.cos(phi_p)], -1)
    sasa = np.zeros(L, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords); mr = r.max()
        for i in range(L):
            nb = [j for j in tree.query_ball_point(coords[i], r[i]+mr) if j != i]
            if not nb: sasa[i] = 1.0; continue
            d = np.linalg.norm((coords[i]+r[i]*sphere)[:, None, :] - coords[nb][None, :, :], axis=-1)
            sasa[i] = np.sum(np.all(d > r[nb][None, :], axis=1)) / n_points
    except ImportError:
        for i in range(L):
            d = np.linalg.norm((coords[i]+r[i]*sphere)[:, None, :] - coords[None, :, :], axis=-1)
            d[:, i] = np.inf; sasa[i] = np.sum(np.all(d > r[None, :], 1)) / n_points
    return np.clip(sasa, 0, 1)
# ── Chain encoding ──
def encode_chain_idx(chain_idx, max_chains=16):
    L = len(chain_idx); oh = np.zeros((L, max_chains), dtype=np.float32)
    for i in range(L): oh[i, min(chain_idx[i], max_chains-1)] = 1.0
    return oh
# ── I/O ──
def save_features(data: dict, path: str):
    np.savez_compressed(path, **{k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in data.items()})
def load_features(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f: return {k: f[k] for k in f.files}
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
    return h.hexdigest()
def discretize_msf(msf, n_bins=32):
    lm = np.log1p(msf); vn, vx = lm.min(), lm.max()
    if vx - vn < 1e-8: return np.zeros_like(msf, dtype=np.int64)
    return np.clip(((lm-vn)/(vx-vn)*n_bins).astype(np.int64), 0, n_bins-1)
def setup_logging(level='INFO'):
    logging.basicConfig(level=getattr(logging, level), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
