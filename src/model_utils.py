"""META model: 4-rank cochain complex, hybrid attn/GCNN, pointer network AR decoding,
cochain+topology masking, torsion biochemistry, implicit multi-chain.
Ranks: 0=residues, 1=edges, 2=bends, 3=torsions."""
import torch, torch.nn as nn, torch.nn.functional as F, math
from typing import Dict, Optional, Tuple, List
from torch import Tensor
# ══════════════════════════════════════════════════════════════════════════════
# Scatter utilities
# ══════════════════════════════════════════════════════════════════════════════
def scatter_softmax_2d(src, index, num_nodes):
    idx = index.unsqueeze(1).expand_as(src)
    mx = torch.full((num_nodes, src.shape[1]), -1e9, device=src.device, dtype=src.dtype)
    mx.scatter_reduce_(0, idx, src, reduce='amax', include_self=True)
    exp_s = (src - mx.gather(0, idx)).exp()
    sm = torch.zeros(num_nodes, src.shape[1], device=src.device, dtype=src.dtype).scatter_add_(0, idx, exp_s)
    return exp_s / (sm.gather(0, idx) + 1e-12)
def scatter_add_3d(src, index, dim_size):
    E, H, D = src.shape
    return torch.zeros(dim_size, H, D, device=src.device, dtype=src.dtype).scatter_add_(0, index.view(E,1,1).expand(E,H,D), src)
def scatter_mean_2d(src, index, dim_size):
    s = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype).scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    c = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype).scatter_add_(0, index.unsqueeze(1), torch.ones(len(index), 1, device=src.device, dtype=src.dtype))
    return s / (c + 1e-8)
# ══════════════════════════════════════════════════════════════════════════════
# Cochain and Topology Masking
# ══════════════════════════════════════════════════════════════════════════════
class CochainMasker(nn.Module):
    def __init__(self, d_dims: List[int]):
        super().__init__()
        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.randn(d) * 0.02) for d in d_dims])
    def forward(self, features, topo, mask_ratio=0.15, topo_mask_ratio=0.0, training=True):
        masked_feats = []; masks = []
        for r, feat in enumerate(features):
            N_r = feat.shape[0]
            if N_r == 0 or mask_ratio <= 0 or not training:
                masked_feats.append(feat); masks.append(torch.zeros(N_r, dtype=torch.bool, device=feat.device)); continue
            n_mask = max(1, int(N_r * mask_ratio))
            perm = torch.randperm(N_r, device=feat.device)[:n_mask]
            mask = torch.zeros(N_r, dtype=torch.bool, device=feat.device); mask[perm] = True
            mf = feat.clone(); mf[mask] = self.mask_tokens[r].to(feat.dtype)
            masked_feats.append(mf); masks.append(mask)
        masked_topo = dict(topo)
        if topo_mask_ratio > 0 and training:
            for sk, dk in [('nbr0_src','nbr0_dst'),('nbr1_src','nbr1_dst'),('nbr2_src','nbr2_dst'),('nbr3_src','nbr3_dst'),
                           ('inc_01_edge','inc_01_node'),('inc_12_bend','inc_12_edge'),('inc_23_torsion','inc_23_bend')]:
                s = topo[sk]; E = len(s)
                if E == 0: continue
                keep = torch.rand(E, device=s.device) > topo_mask_ratio
                masked_topo[sk] = s[keep]; masked_topo[dk] = topo[dk][keep]
        return masked_feats, masks, masked_topo
# ══════════════════════════════════════════════════════════════════════════════
# Reconstruction Heads
# ══════════════════════════════════════════════════════════════════════════════
class FeatureReconHead(nn.Module):
    def __init__(self, d_model, d_feat):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_feat))
    def forward(self, h): return self.mlp(h)
class TopoReconHead(nn.Module):
    def __init__(self, d_model):
        super().__init__(); self.W = nn.Linear(d_model, d_model, bias=False)
    def forward(self, h_i, h_j): return (self.W(h_i) * h_j).sum(-1)
# ══════════════════════════════════════════════════════════════════════════════
# Attention and Convolution layers
# ══════════════════════════════════════════════════════════════════════════════
class SparseNeighbourhoodSelfAttn(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model//n_heads
        self.W_qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.w_bias = nn.Linear(d_model, n_heads, bias=False)
        self.W_gate = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout); self.scale = 1.0/math.sqrt(self.d_k)
    def forward(self, X, nbr_src, nbr_dst, num_cells):
        if len(nbr_src) == 0: return torch.zeros_like(X)
        H, dk = self.n_heads, self.d_k
        qkv = self.W_qkv(X).view(-1, 3, H, dk); Q, K, V = qkv[:,0], qkv[:,1], qkv[:,2]
        gate = torch.sigmoid(self.W_gate(X)).view(-1, H, dk)
        scores = (Q[nbr_src]*K[nbr_dst]).sum(-1)*self.scale + self.w_bias(X)[nbr_src]
        attn = self.dropout(scatter_softmax_2d(scores, nbr_src, num_cells))
        out = scatter_add_3d(attn.unsqueeze(-1)*V[nbr_dst], nbr_src, num_cells)
        return self.W_o((gate*out).reshape(-1, self.d_model))
class SparseGraphConv(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W_self = nn.Linear(d_model, d_model, bias=False); self.W_nbr = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model); self.dropout = nn.Dropout(dropout)
    def forward(self, X, nbr_src, nbr_dst, num_cells):
        if len(nbr_src) == 0: return self.ln(self.W_self(X))
        return self.dropout(self.ln(self.W_self(X) + self.W_nbr(scatter_mean_2d(X[nbr_dst], nbr_src, num_cells))))
class SparseTopologicalCrossAttn(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model//n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False); self.W_kv = nn.Linear(d_model, 2*d_model, bias=False)
        self.w_bias_tgt = nn.Linear(d_model, n_heads, bias=False); self.w_bias_src = nn.Linear(d_model, n_heads, bias=False)
        self.W_gate_tgt = nn.Linear(d_model, d_model); self.W_gate_src = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model, bias=False); self.dropout = nn.Dropout(dropout); self.scale = 1.0/math.sqrt(self.d_k)
    def forward(self, X_tgt, X_src, inc_tgt, inc_src, num_tgt):
        if len(inc_tgt) == 0: return torch.zeros_like(X_tgt)
        H, dk = self.n_heads, self.d_k
        Q = self.W_q(X_tgt).view(-1, H, dk); kv = self.W_kv(X_src).view(-1, 2, H, dk); K, V = kv[:,0], kv[:,1]
        scores = (Q[inc_tgt]*K[inc_src]).sum(-1)*self.scale + self.w_bias_tgt(X_tgt)[inc_tgt] + self.w_bias_src(X_src)[inc_src]
        attn = self.dropout(scatter_softmax_2d(scores, inc_tgt, num_tgt))
        g_src = torch.sigmoid(self.W_gate_src(X_src)).view(-1, H, dk)
        out = scatter_add_3d(attn.unsqueeze(-1)*(g_src[inc_src]*V[inc_src]), inc_tgt, num_tgt)
        return self.W_o((torch.sigmoid(self.W_gate_tgt(X_tgt)).view(-1,H,dk)*out).reshape(-1, self.d_model))
class SparseIncidenceConv(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W_tgt = nn.Linear(d_model, d_model, bias=False); self.W_src = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2*d_model, d_model); self.ln = nn.LayerNorm(d_model); self.dropout = nn.Dropout(dropout)
    def forward(self, X_tgt, X_src, inc_tgt, inc_src, num_tgt):
        if len(inc_tgt) == 0: return self.ln(self.W_tgt(X_tgt))
        agg = scatter_mean_2d(X_src[inc_src], inc_tgt, num_tgt)
        ht = self.W_tgt(X_tgt); hs = self.W_src(agg)
        g = torch.sigmoid(self.gate(torch.cat([ht, hs], -1)))
        return self.dropout(self.ln(g*ht + (1-g)*hs))
class FFN(nn.Module):
    def __init__(self, d_model, d_ff=0, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff or 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff or 4*d_model, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)
# ══════════════════════════════════════════════════════════════════════════════
# META Layer (4 ranks, configurable attn/conv)
# ══════════════════════════════════════════════════════════════════════════════
class METALayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_attention=True):
        super().__init__(); NR = 4
        self.ln_intra = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(NR)])
        self.intra_ops = nn.ModuleList([(SparseNeighbourhoodSelfAttn if use_attention else SparseGraphConv)(d_model, *([n_heads] if use_attention else []), dropout=dropout) for _ in range(NR)])
        self.ln_up_t = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.ln_up_s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.up_ops = nn.ModuleList([(SparseTopologicalCrossAttn if use_attention else SparseIncidenceConv)(d_model, *([n_heads] if use_attention else []), dropout=dropout) for _ in range(3)])
        self.ln_dn_t = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.ln_dn_s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dn_ops = nn.ModuleList([(SparseTopologicalCrossAttn if use_attention else SparseIncidenceConv)(d_model, *([n_heads] if use_attention else []), dropout=dropout) for _ in range(3)])
        self.ln_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(NR)])
        self.ffn = nn.ModuleList([FFN(d_model, dropout=dropout) for _ in range(NR)])
    def forward(self, h, topo):
        N = [x.shape[0] for x in h]
        nk = [('nbr0_src','nbr0_dst'),('nbr1_src','nbr1_dst'),('nbr2_src','nbr2_dst'),('nbr3_src','nbr3_dst')]
        up = [('inc_01_edge','inc_01_node',1),('inc_12_bend','inc_12_edge',2),('inc_23_torsion','inc_23_bend',3)]
        dn = [('inc_23_bend','inc_23_torsion',2),('inc_12_edge','inc_12_bend',1),('inc_01_node','inc_01_edge',0)]
        ht = []
        for r in range(4):
            if N[r] == 0: ht.append(h[r]); continue
            ht.append(h[r] + self.intra_ops[r](self.ln_intra[r](h[r]), topo[nk[r][0]], topo[nk[r][1]], N[r]))
        for i,(tk,sk,tr) in enumerate(up):
            if N[tr]==0 or N[tr-1]==0 or len(topo[tk])==0: continue
            ht[tr] = ht[tr] + self.up_ops[i](self.ln_up_t[i](ht[tr]), self.ln_up_s[i](ht[tr-1]), topo[tk], topo[sk], N[tr])
        for i,(tk,sk,tr) in enumerate(dn):
            if N[tr]==0 or N[tr+1]==0 or len(topo[tk])==0: continue
            ht[tr] = ht[tr] + self.dn_ops[i](self.ln_dn_t[i](ht[tr]), self.ln_dn_s[i](ht[tr+1]), topo[tk], topo[sk], N[tr])
        for r in range(4):
            if N[r]>0: ht[r] = ht[r] + self.ffn[r](self.ln_ffn[r](ht[r]))
        return ht
# ══════════════════════════════════════════════════════════════════════════════
# Pointer Network for learned decoding order
# ══════════════════════════════════════════════════════════════════════════════
class PointerNetwork(nn.Module):
    """Learns optimal decoding order via attention over node embeddings.
    Supports chunked decoding (L positions per step)."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)
        self.gru = nn.GRUCell(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, embeddings, chunk_size=1):
        """Generate decoding order permutation.
        embeddings: (N, d_model). Returns perm: (N,) LongTensor, log_probs: (N/chunk,)."""
        N, d = embeddings.shape; dev = embeddings.device
        proj_e = self.W1(embeddings)  # (N, d) — precompute once
        d_t = embeddings.mean(0)  # (d,) initial hidden state
        selected = torch.zeros(N, dtype=torch.bool, device=dev)
        perm = torch.zeros(N, dtype=torch.long, device=dev)
        log_probs = []
        pos = 0
        while pos < N:
            cs = min(chunk_size, N - pos)
            proj_d = self.W2(d_t)  # (d,)
            scores = self.v(torch.tanh(proj_e + proj_d.unsqueeze(0))).squeeze(-1)  # (N,)
            scores[selected] = -1e9
            probs = F.softmax(scores, dim=0)  # (N,)
            if cs == 1:
                idx = torch.multinomial(probs, 1).squeeze()
                perm[pos] = idx; log_probs.append(torch.log(probs[idx] + 1e-12))
                selected[idx] = True; d_t = self.gru(embeddings[idx].unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
                pos += 1
            else:
                # top-k selection for chunked decoding
                k = min(cs, (~selected).sum().item())
                vals, idxs = probs.topk(k)
                perm[pos:pos+k] = idxs
                log_probs.append(torch.log(vals + 1e-12).sum())
                selected[idxs] = True
                chunk_mean = embeddings[idxs].mean(0)
                d_t = self.gru(chunk_mean.unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
                pos += k
        return perm, torch.stack(log_probs)
# ══════════════════════════════════════════════════════════════════════════════
# Decoder Heads
# ══════════════════════════════════════════════════════════════════════════════
class InputProjection(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__(); self.proj = nn.Linear(d_in, d_model); self.ln = nn.LayerNorm(d_model)
    def forward(self, x): return self.ln(self.proj(x))
class SequenceDecoder(nn.Module):
    def __init__(self, dm, n_aa=20):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, n_aa))
    def forward(self, h0): return self.mlp(h0)
class BiochemDecoder(nn.Module):
    def __init__(self, dm, n_props=4):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(dm, dm//2), nn.GELU(), nn.Linear(dm//2, n_props))
    def forward(self, h): return self.mlp(h)
class MSFDecoder(nn.Module):
    def __init__(self, dm):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(dm, dm//2), nn.GELU(), nn.Linear(dm//2, 1))
    def forward(self, h0): return F.relu(self.mlp(h0)).squeeze(-1)
class PairwiseVarDecoder(nn.Module):
    def __init__(self, dm, dp=64):
        super().__init__()
        self.Wa = nn.Linear(dm,dp,bias=False); self.Wb = nn.Linear(dm,dp,bias=False)
        self.pe = nn.Linear(dm,dp,bias=False); self.mlp = nn.Sequential(nn.Linear(dp,dp//2),nn.GELU(),nn.Linear(dp//2,1))
    def forward(self, h0, h1, es, ed): return F.relu(self.mlp(self.Wa(h0[es])*self.Wb(h0[ed])+self.pe(h1))).squeeze(-1)
class ARDecoder(nn.Module):
    """AR decoder conditioned on MSF, bend context, and torsion biochemistry context."""
    def __init__(self, dm, n_aa=20, n_msf_bins=32, dropout=0.1):
        super().__init__(); self.dm = dm; self.mi = n_aa
        self.ae = nn.Embedding(n_aa+1, dm)
        self.me = nn.Embedding(n_msf_bins, dm//4)
        # conditioning: h0 + msf_emb + bend_ctx + torsion_ctx
        self.cp = nn.Linear(dm + dm//4 + dm + dm, dm)
        self.hd = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, n_aa))
    def forward(self, h0, msf_bins, bend_ctx, torsion_ctx, seq_gt=None, perm=None):
        cond = self.cp(torch.cat([h0, self.me(msf_bins), bend_ctx, torsion_ctx], -1))
        return self.hd(cond)
    def generate(self, h0, msf_bins, bend_ctx, torsion_ctx, perm, temp=1.0, top_p=0.9):
        N = h0.shape[0]
        cond = self.cp(torch.cat([h0, self.me(msf_bins), bend_ctx, torsion_ctx], -1))
        ae = torch.zeros_like(h0); seq = torch.full((N,), self.mi, device=h0.device, dtype=torch.long)
        lp = torch.zeros(N, device=h0.device)
        for t in range(N):
            i = perm[t]; lo = self.hd(cond[i:i+1]+ae[i:i+1]).squeeze(0)/temp; pr = F.softmax(lo, -1)
            sp, si = pr.sort(descending=True); mk = sp.cumsum(-1)-sp > top_p; sp[mk] = 0; sp = sp/(sp.sum()+1e-12)
            cp_ = torch.multinomial(sp, 1); ca = si[cp_]; seq[i] = ca.squeeze()
            lp[i] = torch.log(pr[ca.squeeze()]+1e-12); ae[i] = self.ae(ca.squeeze())
        return seq, lp
# ══════════════════════════════════════════════════════════════════════════════
# parse_layer_types
# ══════════════════════════════════════════════════════════════════════════════
def parse_layer_types(spec, n_layers):
    parts = [s.strip().lower() for s in spec.split(',')]
    if len(parts) == 1: return [parts[0]=='attn']*n_layers
    if len(parts) != n_layers: raise ValueError(f"layer_types has {len(parts)} entries but n_layers={n_layers}")
    return [p=='attn' for p in parts]
# ══════════════════════════════════════════════════════════════════════════════
# Full META Model
# ══════════════════════════════════════════════════════════════════════════════
class METAModel(nn.Module):
    def __init__(self, d_model=32, n_heads=1, n_layers=1, dropout=0.1,
                 d_node=23, d_edge=37, d_bend=1, d_torsion=2,
                 use_ar=False, n_msf_bins=32,
                 layer_types='attn', mask_ratio=0.0, topo_mask_ratio=0.0,
                 use_pointer=False, chunk_size=1):
        super().__init__()
        self.d_model = d_model; self.n_layers = n_layers; self.use_ar = use_ar
        self.mask_ratio = mask_ratio; self.topo_mask_ratio = topo_mask_ratio
        self.use_pointer = use_pointer; self.chunk_size = chunk_size
        self.d_feats = [d_node, d_edge, d_bend, d_torsion]
        # input projections
        self.proj = nn.ModuleList([InputProjection(d, d_model) for d in self.d_feats])
        # masker
        self.masker = CochainMasker(self.d_feats)
        # layers
        lt = parse_layer_types(layer_types, n_layers)
        self.layers = nn.ModuleList([METALayer(d_model, n_heads, dropout, lt[i]) for i in range(n_layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        # primary decoders
        self.seq_decoder = SequenceDecoder(d_model)
        self.biochem_decoder = BiochemDecoder(d_model)  # from h2 (bends)
        self.torsion_biochem_decoder = BiochemDecoder(d_model)  # from h3 (torsions)
        self.msf_decoder = MSFDecoder(d_model)
        self.pair_var_decoder = PairwiseVarDecoder(d_model, max(32, d_model//2))
        # AR decoder + pointer network
        if use_ar:
            self.ar_decoder = ARDecoder(d_model, n_msf_bins=n_msf_bins, dropout=dropout)
            if use_pointer: self.pointer_net = PointerNetwork(d_model, dropout)
        # reconstruction decoders
        self.recon_heads = nn.ModuleList([FeatureReconHead(d_model, d) for d in self.d_feats])
        self.topo_nbr_heads = nn.ModuleList([TopoReconHead(d_model) for _ in range(4)])
        self.topo_inc_heads = nn.ModuleList([TopoReconHead(d_model) for _ in range(3)])
    TOPO_KEYS = ['nbr0_src','nbr0_dst','nbr1_src','nbr1_dst','nbr2_src','nbr2_dst','nbr3_src','nbr3_dst',
                 'inc_01_edge','inc_01_node','inc_12_bend','inc_12_edge','inc_23_torsion','inc_23_bend']
    def _build_topo(self, batch):
        return {k: batch[k] for k in self.TOPO_KEYS}
    def encode(self, features, topo):
        h = [self.proj[r](features[r]) if features[r].shape[0] > 0
             else torch.zeros(0, self.d_model, device=features[0].device, dtype=features[0].dtype) for r in range(4)]
        for layer in self.layers: h = layer(h, topo)
        return [self.ln[r](h[r]) if h[r].shape[0] > 0 else h[r] for r in range(4)]
    def _pool_context(self, h_rank, batch_key, h0, batch):
        """Pool higher-rank latents to node level. h_rank: (M, d), returns (N0, d)."""
        N0 = h0.shape[0]; d = self.d_model; dev = h0.device; dt = h0.dtype
        if h_rank.shape[0] == 0: return torch.zeros(N0, d, device=dev, dtype=dt)
        elements = batch[batch_key]  # (M, 3) for bends or (M, 4) for torsions
        M = elements.shape[0]; ncols = elements.shape[1]
        ctx = torch.zeros(N0, d, device=dev, dtype=dt); cnt = torch.zeros(N0, 1, device=dev, dtype=dt)
        ones = torch.ones(M, 1, device=dev, dtype=dt)
        for c in range(ncols):
            idx = elements[:, c].unsqueeze(1).expand(M, d); ctx.scatter_add_(0, idx, h_rank)
            cnt.scatter_add_(0, elements[:, c].unsqueeze(1), ones)
        return ctx / (cnt + 1e-8)
    def forward(self, batch):
        dev = batch['node_feat'].device
        orig_feats = [batch['node_feat'], batch['edge_feat'], batch['bend_feat'], batch['torsion_feat']]
        orig_topo = self._build_topo(batch)
        mr = self.mask_ratio if self.training else 0.0
        tr = self.topo_mask_ratio if self.training else 0.0
        masked_feats, masks, masked_topo = self.masker(orig_feats, orig_topo, mr, tr, self.training)
        h = self.encode(masked_feats, masked_topo)
        out = {}
        # primary predictions
        out['seq_logits'] = self.seq_decoder(h[0])
        out['biochem_pred'] = self.biochem_decoder(h[2]) if h[2].shape[0] > 0 else torch.zeros(0, 4, device=dev)
        out['torsion_biochem_pred'] = self.torsion_biochem_decoder(h[3]) if h[3].shape[0] > 0 else torch.zeros(0, 4, device=dev)
        out['msf_pred'] = self.msf_decoder(h[0])
        out['pair_var_pred'] = self.pair_var_decoder(h[0], h[1], batch['edge_src'], batch['edge_dst']) if h[1].shape[0] > 0 else torch.zeros(0, device=dev)
        # AR decoder with bend + torsion context
        if self.use_ar and hasattr(self, 'ar_decoder'):
            bend_ctx = self._pool_context(h[2], 'bends', h[0], batch)
            torsion_ctx = self._pool_context(h[3], 'torsions', h[0], batch)
            msf_bins = self._discretize_msf(out['msf_pred'].detach())
            # pointer network for decoding order
            if self.use_pointer and hasattr(self, 'pointer_net'):
                perm, ptr_log_probs = self.pointer_net(h[0].detach(), self.chunk_size)
                out['ptr_log_probs'] = ptr_log_probs; out['perm'] = perm
            out['ar_logits'] = self.ar_decoder(h[0], msf_bins, bend_ctx, torsion_ctx, batch['seq_idx'], batch.get('node_batch'))
        # feature reconstruction
        out['recon_preds'] = []; out['recon_targets'] = []; out['recon_masks'] = masks
        for r in range(4):
            if h[r].shape[0] > 0 and masks[r].any():
                out['recon_preds'].append(self.recon_heads[r](h[r][masks[r]]))
                out['recon_targets'].append(orig_feats[r][masks[r]])
            else:
                out['recon_preds'].append(torch.zeros(0, self.d_feats[r], device=dev))
                out['recon_targets'].append(torch.zeros(0, self.d_feats[r], device=dev))
        # topology reconstruction
        out['topo_nbr_logits'] = []; out['topo_nbr_labels'] = []
        out['topo_inc_logits'] = []; out['topo_inc_labels'] = []
        if tr > 0 and self.training:
            nbr_pairs = [('nbr0_src','nbr0_dst',0),('nbr1_src','nbr1_dst',1),('nbr2_src','nbr2_dst',2),('nbr3_src','nbr3_dst',3)]
            for sk, dk, r in nbr_pairs:
                n_pos = len(orig_topo[sk]); Nr = h[r].shape[0]
                if n_pos == 0 or Nr < 2:
                    out['topo_nbr_logits'].append(torch.zeros(0, device=dev)); out['topo_nbr_labels'].append(torch.zeros(0, device=dev)); continue
                ps, pd = orig_topo[sk], orig_topo[dk]; mp = min(n_pos, 2000)
                if n_pos > mp: idx = torch.randperm(n_pos, device=dev)[:mp]; ps, pd = ps[idx], pd[idx]
                pl = self.topo_nbr_heads[r](h[r][ps], h[r][pd])
                ns, nd = torch.randint(0, Nr, (len(ps),), device=dev), torch.randint(0, Nr, (len(ps),), device=dev)
                nl = self.topo_nbr_heads[r](h[r][ns], h[r][nd])
                out['topo_nbr_logits'].append(torch.cat([pl, nl])); out['topo_nbr_labels'].append(torch.cat([torch.ones_like(pl), torch.zeros_like(nl)]))
            inc_pairs = [('inc_01_edge','inc_01_node',1,0),('inc_12_bend','inc_12_edge',2,1),('inc_23_torsion','inc_23_bend',3,2)]
            for tk, sk, rh, rl in inc_pairs:
                n_pos = len(orig_topo[tk]); Nh, Nl = h[rh].shape[0], h[rl].shape[0]
                if n_pos == 0 or Nh < 1 or Nl < 1:
                    out['topo_inc_logits'].append(torch.zeros(0, device=dev)); out['topo_inc_labels'].append(torch.zeros(0, device=dev)); continue
                pt, ps = orig_topo[tk], orig_topo[sk]; mp = min(n_pos, 2000)
                if n_pos > mp: idx = torch.randperm(n_pos, device=dev)[:mp]; pt, ps = pt[idx], ps[idx]
                pl = self.topo_inc_heads[rl](h[rh][pt], h[rl][ps])
                nt, ns = torch.randint(0, Nh, (len(pt),), device=dev), torch.randint(0, Nl, (len(pt),), device=dev)
                nl = self.topo_inc_heads[rl](h[rh][nt], h[rl][ns])
                out['topo_inc_logits'].append(torch.cat([pl, nl])); out['topo_inc_labels'].append(torch.cat([torch.ones_like(pl), torch.zeros_like(nl)]))
        return out
    @staticmethod
    def _discretize_msf(msf, n_bins=32):
        lm = torch.log1p(msf); vn, vx = lm.min(), lm.max()
        if vx-vn < 1e-8: return torch.zeros_like(msf, dtype=torch.long)
        return ((lm-vn)/(vx-vn+1e-8)*n_bins).long().clamp(0, n_bins-1)
# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════
class LabelSmoothedCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__(); self.s = smoothing
    def forward(self, logits, targets, mask=None):
        lp = F.log_softmax(logits, -1)
        loss = (1-self.s)*F.nll_loss(lp, targets, reduction='none') + self.s*(-lp.mean(-1))
        if mask is not None: loss = loss*mask.float()
        return loss.sum()/(mask.float().sum()+1e-8) if mask is not None else loss.mean()
class METALoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.0, delta=0.1, zeta=0.1, smoothing=0.1):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta, self.zeta = alpha, beta, gamma, delta, zeta
        self.seq_loss = LabelSmoothedCE(smoothing)
    def forward(self, pred, batch, use_ar=False):
        losses = {}; dev = pred['seq_logits'].device
        sm = batch['seq_idx'] < 20; st = batch['seq_idx'].clamp(0, 19)
        if use_ar and 'ar_logits' in pred:
            l_seq = self.seq_loss(pred['ar_logits'], st, sm); losses['ar_loss'] = l_seq.item()
        else:
            l_seq = self.seq_loss(pred['seq_logits'], st, sm); losses['seq_loss'] = l_seq.item()
        # bend + torsion biochemistry
        l_bc = torch.tensor(0., device=dev)
        if pred['biochem_pred'].shape[0] > 0:
            l_bc = l_bc + F.mse_loss(pred['biochem_pred'], batch['biochem_targets'])
        if pred['torsion_biochem_pred'].shape[0] > 0:
            l_bc = l_bc + F.mse_loss(pred['torsion_biochem_pred'], batch['torsion_biochem_targets'])
        losses['biochem_loss'] = l_bc.item()
        # dynamics
        has = batch['has_dynamics']
        if self.gamma > 0 and has.any():
            dm = has[batch['node_batch']].float()
            l_msf = ((pred['msf_pred']-batch['msf'])**2*dm).sum()/(dm.sum()+1e-8) if dm.sum()>0 else torch.tensor(0.,device=dev)
            if pred['pair_var_pred'].shape[0] > 0:
                de = has[batch['edge_batch']].float()
                l_pv = ((pred['pair_var_pred']-batch['pair_var'])**2*de).sum()/(de.sum()+1e-8) if de.sum()>0 else torch.tensor(0.,device=dev)
            else: l_pv = torch.tensor(0.,device=dev)
            l_dyn = l_msf + l_pv; losses['msf_loss'] = l_msf.item(); losses['pair_var_loss'] = l_pv.item()
        else:
            l_dyn = torch.tensor(0.,device=dev); losses['msf_loss'] = 0.; losses['pair_var_loss'] = 0.
        # feature reconstruction
        l_recon = torch.tensor(0., device=dev); n_masked = 0
        if 'recon_preds' in pred:
            for r in range(4):
                p, t = pred['recon_preds'][r], pred['recon_targets'][r]
                if p.shape[0] > 0: l_recon = l_recon + F.mse_loss(p, t)*p.shape[0]; n_masked += p.shape[0]
            if n_masked > 0: l_recon = l_recon / n_masked
        losses['recon_loss'] = l_recon.item()
        # topology reconstruction
        l_topo = torch.tensor(0., device=dev); n_topo = 0
        for key in ['topo_nbr_logits', 'topo_inc_logits']:
            if key not in pred: continue
            lk = key.replace('logits', 'labels')
            for lo, la in zip(pred[key], pred[lk]):
                if lo.shape[0] > 0: l_topo = l_topo + F.binary_cross_entropy_with_logits(lo, la)*lo.shape[0]; n_topo += lo.shape[0]
        if n_topo > 0: l_topo = l_topo / n_topo
        losses['topo_loss'] = l_topo.item()
        total = self.alpha*l_seq + self.beta*l_bc + self.gamma*l_dyn + self.delta*l_recon + self.zeta*l_topo
        losses['total'] = total.item()
        return total, losses
