"""META training utilities: curriculum with chunk annealing, scheduled sampling,
warmup cosine scheduler, early stopping, metrics, checkpointing."""
import logging, time, os
logger = logging.getLogger(__name__)
# ── Parameter counting ──
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
# ── Device transfer ──
def to_device(batch, device):
    import torch
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
# ── FIX #3 + #4: Training curriculum with chunk annealing + scheduled sampling ──
class TrainingCurriculum:
    """4-phase curriculum per proposal:
    P1 (1..p1): encoder warm-up, parallel seq + biochem, gamma=0
    P2 (p1+1..p1+p2): ramp gamma linearly to target
    P3 (p1+p2+1..p1+p2+p3): AR + pointer, chunk L anneals from N/4 to 1
    P4 (>p1+p2+p3): fine-tune, L=1, scheduled sampling at sched_sample_ratio"""
    def __init__(self, p1=20, p2=30, p3=50, gamma_target=0.5, sched_sample_ratio=0.2):
        self.p1 = p1; self.p2 = p2; self.p3 = p3
        self.gamma_target = gamma_target; self.sched_sample_ratio = sched_sample_ratio
    def get_phase(self, epoch):
        if epoch < self.p1: return 1
        if epoch < self.p1 + self.p2: return 2
        if epoch < self.p1 + self.p2 + self.p3: return 3
        return 4
    def get_gamma(self, epoch):
        if epoch < self.p1: return 0.0
        if epoch < self.p1 + self.p2:
            frac = (epoch - self.p1) / max(1, self.p2)
            return self.gamma_target * frac
        return self.gamma_target
    def use_ar(self, epoch):
        return self.get_phase(epoch) >= 3
    def get_chunk_size(self, epoch, max_len=500):
        """FIX #3: chunk anneals from max_len//4 to 1 during Phase 3."""
        phase = self.get_phase(epoch)
        if phase < 3: return max_len  # parallel decoding
        if phase == 3:
            progress = (epoch - self.p1 - self.p2) / max(1, self.p3)
            start_chunk = max(max_len // 4, 2)
            chunk = int(start_chunk * (1.0 - progress) + 1.0 * progress)
            return max(1, chunk)
        return 1  # phase 4: fully sequential
    def get_sched_sample_ratio(self, epoch):
        """FIX #4: scheduled sampling ratio — only active in Phase 4."""
        if self.get_phase(epoch) >= 4: return self.sched_sample_ratio
        return 0.0
    def __repr__(self):
        return f"Curriculum(P1={self.p1}, P2={self.p2}, P3={self.p3}, gamma={self.gamma_target})"
# ── Warmup Cosine LR Scheduler ──
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.opt = optimizer; self.warmup = warmup_steps; self.total = total_steps; self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]; self.step_count = 0
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup:
            frac = self.step_count / max(1, self.warmup)
        else:
            progress = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            frac = 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))
        for g, base_lr in zip(self.opt.param_groups, self.base_lrs):
            g['lr'] = max(self.min_lr, base_lr * frac)
    def state_dict(self): return {'step_count': self.step_count}
    def load_state_dict(self, sd): self.step_count = sd['step_count']
# ── Early Stopping ──
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience; self.counter = 0; self.best = float('inf'); self.should_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best: self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.should_stop = True
        return self.should_stop
# ── Timer ──
class Timer:
    def __init__(self): self.start = time.time()
    def elapsed(self): return time.time() - self.start
    def reset(self): self.start = time.time()
# ── Metrics ──
def compute_all_metrics(pred, batch, use_ar=False):
    import torch; import torch.nn.functional as F
    m = {}
    logits = pred['ar_logits'] if (use_ar and 'ar_logits' in pred) else pred['seq_logits']
    pred_aa = logits.argmax(-1); gt = batch['seq_idx'].clamp(0, 19)
    valid = batch['seq_idx'] < 20
    m['recovery'] = ((pred_aa == gt) & valid).float().sum().item() / (valid.float().sum().item() + 1e-8)
    if 'msf_pred' in pred and batch['msf'].shape[0] > 0:
        p = pred['msf_pred'].detach(); t = batch['msf']
        if p.shape[0] == t.shape[0] and t.std() > 1e-6 and p.std() > 1e-6:
            m['msf_pearson'] = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
    # reconstruction metrics
    if 'recon_preds' in pred:
        total_masked = sum(p.shape[0] for p in pred['recon_preds'])
        if total_masked > 0:
            recon_mse = sum(F.mse_loss(p, t).item()*p.shape[0] for p, t in zip(pred['recon_preds'], pred['recon_targets']) if p.shape[0]>0) / total_masked
            m['recon_mse'] = recon_mse
    if 'topo_nbr_logits' in pred:
        all_l = torch.cat([l for l in pred['topo_nbr_logits'] if l.shape[0]>0] + [l for l in pred.get('topo_inc_logits',[]) if l.shape[0]>0])
        all_t = torch.cat([l for l in pred['topo_nbr_labels'] if l.shape[0]>0] + [l for l in pred.get('topo_inc_labels',[]) if l.shape[0]>0])
        if all_l.shape[0] > 0: m['topo_acc'] = ((all_l > 0).float() == all_t).float().mean().item()
    return m
# ── Checkpointing ──
def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics, path, scaler=None):
    import torch
    st = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch, 'step': step, 'metrics': metrics}
    if scheduler: st['scheduler_state'] = scheduler.state_dict()
    if scaler: st['scaler_state'] = scaler.state_dict()
    torch.save(st, path); logger.info(f"Saved: {path} (epoch={epoch})")
def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state'])
    if optimizer and 'optimizer_state' in ck: optimizer.load_state_dict(ck['optimizer_state'])
    if scheduler and 'scheduler_state' in ck: scheduler.load_state_dict(ck['scheduler_state'])
    if scaler and 'scaler_state' in ck: scaler.load_state_dict(ck['scaler_state'])
    return {'epoch': ck['epoch'], 'step': ck['step'], 'metrics': ck.get('metrics', {})}
