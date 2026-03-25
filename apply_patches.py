"""apply_patches.py — Automated patcher for META fixes #1, #6, #7.
Run from the repo root: python apply_patches.py
This modifies src/model_utils.py, src/processing.py, and train.py in-place.
Creates .bak backups before modifying."""
import re, os, sys, shutil
def backup_and_read(path):
    if not os.path.exists(path): print(f"ERROR: {path} not found"); sys.exit(1)
    shutil.copy2(path, path + '.bak')
    with open(path) as f: return f.read()
def write(path, content):
    with open(path, 'w') as f: f.write(content)
    print(f"  Patched: {path}")
def patch_model_utils():
    """Patch src/model_utils.py for fixes #1 and #6."""
    path = 'src/model_utils.py'
    code = backup_and_read(path)
    # ── Fix #1: ARDecoder.forward() — add sched_sample_ratio ──
    old_sig = 'def forward(self, h0, msf_bins, bend_ctx, torsion_ctx, seq_gt, perm, node_batch=None):'
    new_sig = 'def forward(self, h0, msf_bins, bend_ctx, torsion_ctx, seq_gt, perm, node_batch=None, sched_sample_ratio=0.0):'
    if old_sig in code:
        code = code.replace(old_sig, new_sig, 1)
        print("  [Fix#1a] ARDecoder.forward() signature updated")
    else:
        print("  WARN: ARDecoder.forward() signature not found — may already be patched")
    # ── Fix #1: add scheduled sampling logic before h_with_aa ──
    old_inject = '        h_with_aa = cond + self.aa_emb(seq_gt.clamp(0, self.n_aa - 1))  # (N, dm)'
    new_inject = """        # FIX #1: scheduled sampling — replace fraction of GT with model predictions in Phase 4
        if sched_sample_ratio > 0 and self.training:
            with torch.no_grad():
                h_tmp = cond.unsqueeze(0)
                no_mask = torch.zeros(N, N, device=dev, dtype=cond.dtype)
                pred_logits = self.head(self.transformer_dec(h_tmp, h_tmp, tgt_mask=no_mask, memory_mask=no_mask).squeeze(0))
                pred_aa = pred_logits.argmax(-1)
            ss_mask = torch.rand(N, device=dev) < sched_sample_ratio
            seq_input = torch.where(ss_mask, pred_aa.clamp(0, self.n_aa-1), seq_gt.clamp(0, self.n_aa-1))
        else:
            seq_input = seq_gt.clamp(0, self.n_aa - 1)
        h_with_aa = cond + self.aa_emb(seq_input)  # (N, dm)"""
    if old_inject in code:
        code = code.replace(old_inject, new_inject, 1)
        print("  [Fix#1b] Scheduled sampling logic inserted")
    else:
        print("  WARN: h_with_aa injection point not found")
    # ── Fix #1: METAModel.forward() signature ──
    old_fwd = 'def forward(self, batch):'
    new_fwd = 'def forward(self, batch, sched_sample_ratio=0.0):'
    # only replace the one in METAModel (should be the last one or the one after class METAModel)
    if code.count(old_fwd) >= 1:
        # replace last occurrence (METAModel's forward, not other classes)
        idx = code.rfind(old_fwd)
        code = code[:idx] + new_fwd + code[idx + len(old_fwd):]
        print("  [Fix#1c] METAModel.forward() signature updated")
    # ── Fix #1: pass sched_sample_ratio to ar_decoder ──
    old_ar_call = "out['ar_logits'] = self.ar_decoder(h[0], msf_bins, bend_ctx, torsion_ctx, batch['seq_idx'], perm, nb)"
    new_ar_call = "out['ar_logits'] = self.ar_decoder(h[0], msf_bins, bend_ctx, torsion_ctx, batch['seq_idx'], perm, nb, sched_sample_ratio)"
    if old_ar_call in code:
        code = code.replace(old_ar_call, new_ar_call, 1)
        print("  [Fix#1d] ar_decoder call updated with sched_sample_ratio")
    else:
        print("  WARN: ar_decoder call pattern not found")
    # ── Fix #6: skip seq_decoder when AR is active during training ──
    old_seq = "out['seq_logits'] = self.seq_decoder(h[0])"
    new_seq = """# FIX #6: skip one-shot seq_decoder when AR active during training (save compute)
        if self.use_ar and hasattr(self, 'ar_decoder') and self.training:
            out['seq_logits'] = torch.zeros(h[0].shape[0], 20, device=dev)
        else:
            out['seq_logits'] = self.seq_decoder(h[0])"""
    if old_seq in code:
        code = code.replace(old_seq, new_seq, 1)
        print("  [Fix#6] seq_decoder skip logic inserted")
    else:
        print("  WARN: seq_logits line not found")
    write(path, code)
def patch_processing():
    """Patch src/processing.py for fix #7."""
    path = 'src/processing.py'
    code = backup_and_read(path)
    old_dir = 'dir_raw = CA[edst] - CA[esrc]'
    new_dir = 'dir_raw = CB[edst] - CB[esrc]  # FIX #7: use CB (not CA) to match paper edge definition'
    if old_dir in code:
        code = code.replace(old_dir, new_dir, 1)
        print("  [Fix#7] Direction vector changed from CA to CB")
    else:
        print("  WARN: CA direction line not found — may already be patched")
    write(path, code)
def patch_train():
    """Patch train.py for fix #1 (wire scheduled sampling)."""
    path = 'train.py'
    code = backup_and_read(path)
    # Add sched_ratio computation after chunk_now
    old_chunk = """        chunk_now = curriculum.get_chunk_size(epoch, args.max_len)
        if hasattr(model, 'chunk_size'): model.chunk_size = chunk_now
        ep_losses = []; t0 = time.time()"""
    new_chunk = """        chunk_now = curriculum.get_chunk_size(epoch, args.max_len)
        if hasattr(model, 'chunk_size'): model.chunk_size = chunk_now
        sched_ratio = curriculum.get_sched_sample_ratio(epoch)  # FIX #1: scheduled sampling
        ep_losses = []; t0 = time.time()"""
    if old_chunk in code:
        code = code.replace(old_chunk, new_chunk, 1)
        print("  [Fix#1e] sched_ratio computation added")
    else:
        print("  WARN: chunk_now block not found")
    # Replace model(batch) with model(batch, sched_ratio) in both branches
    # mixed precision branch
    code = code.replace(
        'pred = model(batch); loss, ld = loss_fn(pred, batch, use_ar_now)',
        'pred = model(batch, sched_ratio); loss, ld = loss_fn(pred, batch, use_ar_now)')
    print("  [Fix#1f] model() calls updated with sched_ratio")
    write(path, code)
def main():
    print("=" * 60)
    print("META Patcher — Applying Fixes #1, #6, #7")
    print("=" * 60)
    if not os.path.exists('src/model_utils.py'):
        print("ERROR: Run this from the repo root (where src/ is)"); sys.exit(1)
    print("\n[1/3] Patching src/model_utils.py (Fix #1 + #6)...")
    patch_model_utils()
    print("\n[2/3] Patching src/processing.py (Fix #7)...")
    patch_processing()
    print("\n[3/3] Patching train.py (Fix #1)...")
    patch_train()
    print("\n" + "=" * 60)
    print("Done. Backups saved as *.bak files.")
    print("New files to copy manually: inference.py, validate_bfactors.py")
    print("=" * 60)
if __name__ == '__main__': main()
