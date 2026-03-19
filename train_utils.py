"""META training utilities: metrics, checkpointing, curriculum, scheduling."""
import torch, torch.nn.functional as F, os, logging, math, time
import numpy as np
from typing import Dict, Optional
from scipy.stats import pearsonr
logger = logging.getLogger(__name__)
# ── Metrics ──
def sequence_recovery(logits, targets, mask=None):
    preds = logits.argmax(-1); correct = preds == targets
    if mask is not None: correct = correct & mask
    n = mask.sum().item() if mask is not None else len(targets)
    return correct.sum().item() / max(n, 1)
def perplexity(logits, targets, mask=None):
    lp = torch.nn.functional.log_softmax(logits, -1)
    nll = torch.nn.functional.nll_loss(lp, targets, reduction='none')
    if mask is not None: nll = nll * mask.float()
    avg = nll.sum() / (mask.float().sum()+1e-8) if mask is not None else nll.mean()
    return torch.exp(avg).item()
def msf_correlation(pred, target, batch_idx, has_dyn):
    corrs = []
    for b in range(has_dyn.shape[0]):
        if has_dyn[b].item() == 0: continue
        m = batch_idx == b; p = pred[m].detach().cpu().numpy(); t = target[m].detach().cpu().numpy()
        if len(p) < 3 or t.std() < 1e-8: continue
        r, _ = pearsonr(p, t)
        if not np.isnan(r): corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0
def compute_all_metrics(pred, batch, use_ar=False):
    m = {}; seq_mask = batch['seq_idx'] < 20; seq_tgt = batch['seq_idx'].clamp(0, 19)
    lk = 'ar_logits' if (use_ar and 'ar_logits' in pred) else 'seq_logits'
    m['recovery'] = sequence_recovery(pred[lk], seq_tgt, seq_mask)
    m['perplexity'] = perplexity(pred[lk], seq_tgt, seq_mask)
    if pred['biochem_pred'].shape[0] > 0:
        m['biochem_mae'] = (pred['biochem_pred'] - batch['biochem_targets']).abs().mean().item()
    if batch['has_dynamics'].any():
        m['msf_corr'] = msf_correlation(pred['msf_pred'], batch['msf'], batch['node_batch'], batch['has_dynamics'])
    # reconstruction metrics
    if 'recon_preds' in pred:
        total_masked = sum(p.shape[0] for p in pred['recon_preds'])
        if total_masked > 0:
            recon_mse = sum(F.mse_loss(p, t).item()*p.shape[0] for p, t in zip(pred['recon_preds'], pred['recon_targets']) if p.shape[0]>0) / total_masked
            m['recon_mse'] = recon_mse
    if 'topo_nbr_logits' in pred:
        all_l = torch.cat([l for l in pred['topo_nbr_logits'] if l.shape[0]>0] + [l for l in pred.get('topo_inc_logits',[]) if l.shape[0]>0])
        all_t = torch.cat([l for l in pred['topo_nbr_labels'] if l.shape[0]>0] + [l for l in pred.get('topo_inc_labels',[]) if l.shape[0]>0])
        if all_l.shape[0] > 0:
            m['topo_auroc'] = ((all_l > 0).float() == all_t).float().mean().item()  # approx accuracy
    return m
# ── Checkpointing ──
def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics, path, scaler=None):
    st = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch, 'step': step, 'metrics': metrics}
    if scheduler: st['scheduler_state'] = scheduler.state_dict()
    if scaler: st['scaler_state'] = scaler.state_dict()
    torch.save(st, path); logger.info(f"Saved: {path} (epoch={epoch})")
def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state'])
    if optimizer and 'optimizer_state' in ck: optimizer.load_state_dict(ck['optimizer_state'])
    if scheduler and 'scheduler_state' in ck: scheduler.load_state_dict(ck['scheduler_state'])
    if scaler and 'scaler_state' in ck: scaler.load_state_dict(ck['scaler_state'])
    return {'epoch': ck['epoch'], 'step': ck['step'], 'metrics': ck.get('metrics', {})}
# ── Curriculum ──
class TrainingCurriculum:
    def __init__(self, p1=20, p2=30, p3=50, gamma_target=0.5, ss_max=0.2):
        self.p1e = p1; self.p2e = p1+p2; self.p3e = p1+p2+p3; self.gt = gamma_target; self.ssm = ss_max
    def get_phase(self, ep):
        if ep < self.p1e: return 1
        return 2 if ep < self.p2e else 3
    def get_gamma(self, ep):
        p = self.get_phase(ep)
        if p == 1: return 0.0
        if p == 2: return self.gt * (ep-self.p1e)/max(1, self.p2e-self.p1e)
        return self.gt
    def use_ar(self, ep): return self.get_phase(ep) == 3
    def __repr__(self): return f"Curriculum(P1=0-{self.p1e}, P2={self.p1e}-{self.p2e}, P3={self.p2e}-{self.p3e})"
# ── LR Scheduler ──
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_lr=1e-6, last_epoch=-1):
        self.ws = warmup; self.ts = total; self.ml = min_lr; super().__init__(opt, last_epoch)
    def get_lr(self):
        s = self.last_epoch
        sc = s/max(1, self.ws) if s < self.ws else 0.5*(1+math.cos(math.pi*(s-self.ws)/max(1, self.ts-self.ws)))
        return [max(self.ml, lr*sc) for lr in self.base_lrs]
# ── Early Stopping ──
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.p = patience; self.md = min_delta; self.best = None; self.c = 0; self.stop = False
    def step(self, v):
        if self.best is None: self.best = v; return False
        if v < self.best - self.md: self.best = v; self.c = 0
        else:
            self.c += 1
            if self.c >= self.p: self.stop = True
        return self.stop
class Timer:
    def __init__(self): self.t0 = time.time()
    def total(self): return time.time() - self.t0
def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
