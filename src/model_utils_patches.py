"""
╔══════════════════════════════════════════════════════════════════════╗
║  PATCHES for src/model_utils.py — Fixes #1 (scheduled sampling),   ║
║  #6 (skip seq_decoder when AR active)                               ║
║  Apply these changes to your existing src/model_utils.py            ║
╚══════════════════════════════════════════════════════════════════════╝

PATCH 1 — ARDecoder.forward(): add scheduled sampling support
=====================================================================
Replace the ENTIRE ARDecoder.forward() method with the version below.
The key change: when sched_sample_ratio > 0, a fraction of positions
use the model's own argmax predictions instead of ground-truth tokens
during teacher forcing — exactly as §2.6 Phase 4 specifies.
"""

# ── REPLACE ARDecoder.forward() with this ──────────────────────────────────
def forward_PATCHED(self, h0, msf_bins, bend_ctx, torsion_ctx, seq_gt, perm,
                    node_batch=None, sched_sample_ratio=0.0):
    """Teacher-forced AR with optional scheduled sampling (Phase 4).
    sched_sample_ratio: fraction of positions that use model predictions
    instead of ground-truth during teacher forcing. 0.0 = pure teacher forcing."""
    import torch, torch.nn.functional as F
    N, dm, dev = h0.shape[0], self.dm, h0.device
    # build conditioning vector per proposal eq
    cond = self.cond_proj(torch.cat([h0, self.msf_emb(msf_bins), bend_ctx, torsion_ctx], -1))
    # scheduled sampling: replace some GT tokens with model predictions
    if sched_sample_ratio > 0 and self.training:
        with torch.no_grad():
            # get one-shot predictions from conditioning alone (no AA context)
            h_no_aa = cond.unsqueeze(0)
            no_mask = torch.zeros(N, N, device=dev, dtype=cond.dtype)  # attend everywhere
            pred_logits = self.head(self.transformer_dec(h_no_aa, h_no_aa, tgt_mask=no_mask, memory_mask=no_mask).squeeze(0))
            pred_aa = pred_logits.argmax(-1)
        # build scheduled sampling mask: True = use model prediction
        ss_mask = torch.rand(N, device=dev) < sched_sample_ratio
        seq_input = torch.where(ss_mask, pred_aa.clamp(0, self.n_aa - 1), seq_gt.clamp(0, self.n_aa - 1))
    else:
        seq_input = seq_gt.clamp(0, self.n_aa - 1)
    # inject AA embeddings into node states (proposal eq)
    h_with_aa = cond + self.aa_emb(seq_input)
    # causal mask from decoding order
    causal_mask = self._make_causal_mask(perm, N, dev, cond.dtype, node_batch)
    query = cond.unsqueeze(0); memory = h_with_aa.unsqueeze(0)
    decoded = self.transformer_dec(query, memory, tgt_mask=causal_mask, memory_mask=causal_mask)
    return self.head(decoded.squeeze(0))


"""
PATCH 2 — METAModel.forward(): accept sched_sample_ratio, skip seq_decoder when AR active
==============================================================================================
In the METAModel.forward() method, apply three changes:

A) Add sched_sample_ratio parameter to forward() signature
B) Pass it through to ar_decoder()
C) Skip seq_decoder when AR is active (save compute in phases 3-4)

Find this section in METAModel.forward():
"""

# ── ORIGINAL (find this block in your forward method) ──────────────────────
# out['seq_logits'] = self.seq_decoder(h[0])
# ...
# if self.use_ar and hasattr(self, 'ar_decoder'):
#     ...
#     out['ar_logits'] = self.ar_decoder(h[0], msf_bins, bend_ctx, torsion_ctx, batch['seq_idx'], perm, nb)

# ── REPLACE WITH ───────────────────────────────────────────────────────────
def forward_PATCHED_model(self, batch, sched_sample_ratio=0.0):
    """
    In the existing forward(), make these 3 surgical changes:

    1) Change the method signature to accept sched_sample_ratio:
       def forward(self, batch, sched_sample_ratio=0.0):

    2) Replace the seq_logits + AR block with:
    """
    # --- FIX #6: skip one-shot head when AR is active to save compute ---
    if self.use_ar and hasattr(self, 'ar_decoder') and self.training:
        # still need seq_logits shape for loss_fn fallback; use zeros (no grad, no compute)
        out['seq_logits'] = torch.zeros(h[0].shape[0], 20, device=dev)
    else:
        out['seq_logits'] = self.seq_decoder(h[0])
    # AR decoder with conditioning
    if self.use_ar and hasattr(self, 'ar_decoder'):
        bend_ctx = self._pool_context(h[2], 'bends', h[0], batch)
        torsion_ctx = self._pool_context(h[3], 'torsions', h[0], batch)
        nb = batch.get('node_batch') if self.per_protein_ar else None
        msf_bins = self._discretize_msf(out['msf_pred'].detach(), nb)
        # build permutation (pointer or random)
        if self.use_pointer and hasattr(self, 'pointer_net'):
            if self.per_protein_ar:
                perm, ptr_log_probs = self._run_pointer_per_protein(h[0], nb)
            else:
                perm, ptr_log_probs = self.pointer_net(h[0], self.chunk_size)
            out['ptr_log_probs'] = ptr_log_probs; out['perm'] = perm
        else:
            if self.per_protein_ar:
                perm = self._random_perm_per_protein(h[0].shape[0], nb, dev)
            else:
                perm = torch.randperm(h[0].shape[0], device=dev)
            out['perm'] = perm
        # FIX #1: pass sched_sample_ratio to AR decoder
        out['ar_logits'] = self.ar_decoder(h[0], msf_bins, bend_ctx, torsion_ctx,
                                           batch['seq_idx'], perm, nb, sched_sample_ratio)
    pass  # rest of forward unchanged


"""
PATCH 3 — train.py training loop: wire scheduled sampling from curriculum
==========================================================================
In the training loop (the per-epoch block), add one line to get the
scheduled sampling ratio, and pass it to model().

Find this section in train.py:
"""

# ── ORIGINAL (in train.py, inside the epoch loop) ─────────────────────────
# chunk_now = curriculum.get_chunk_size(epoch, args.max_len)
# if hasattr(model, 'chunk_size'): model.chunk_size = chunk_now
# ...
# pred = model(batch)

# ── REPLACE WITH ──────────────────────────────────────────────────────────
# After the chunk_now line, add:
#   sched_ratio = curriculum.get_sched_sample_ratio(epoch)
#
# Then change every call of model(batch) to model(batch, sched_ratio):
#   pred = model(batch, sched_ratio)
#
# This applies to BOTH the mixed_precision and non-mixed_precision branches.
# Full replacement of the relevant training loop block:

"""
        # FIX #3: dynamic chunk size from curriculum
        chunk_now = curriculum.get_chunk_size(epoch, args.max_len)
        if hasattr(model, 'chunk_size'): model.chunk_size = chunk_now
        # FIX #1: scheduled sampling ratio from curriculum (active only in Phase 4)
        sched_ratio = curriculum.get_sched_sample_ratio(epoch)
        ep_losses = []; t0 = time.time()
        for si, batch in enumerate(train_dl):
            batch = to_device(batch, device); optimizer.zero_grad(set_to_none=True)
            if args.mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(batch, sched_ratio); loss, ld = loss_fn(pred, batch, use_ar_now)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(batch, sched_ratio); loss, ld = loss_fn(pred, batch, use_ar_now)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
"""
