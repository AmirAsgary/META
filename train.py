"""META Training Pipeline. Usage: python train.py --data_dir ./pdbs --output_dir ./out
Example with hybrid layers: --n_layers 4 --layer_types attn,conv,conv,conv"""
import argparse, os, sys, logging, time, json, torch, numpy as np
from pathlib import Path
def parse_args():
    p = argparse.ArgumentParser(description='META training')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--output_dir', type=str, default='./output')
    p.add_argument('--split_file', type=str, default=None)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--max_len', type=int, default=500)
    p.add_argument('--min_len', type=int, default=30)
    p.add_argument('--edge_cutoff', type=float, default=8.0)
    p.add_argument('--compute_dynamics', action='store_true')
    p.add_argument('--n_modes', type=int, default=20)
    p.add_argument('--n_conformers', type=int, default=1000)
    p.add_argument('--n_workers_process', type=int, default=4)
    p.add_argument('--per_chain', action='store_true', default=True)
    p.add_argument('--no_per_chain', dest='per_chain', action='store_false')
    # architecture
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--n_heads', type=int, default=1)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--layer_types', type=str, default='attn', help='Comma-separated: attn or conv per layer. E.g. attn,conv,conv,conv')
    p.add_argument('--use_ar', action='store_true')
    p.add_argument('--use_pointer', action='store_true', help='Use pointer network for learned decoding order')
    p.add_argument('--chunk_size', type=int, default=1, help='Positions decoded per pointer step (1=sequential)')
    p.add_argument('--n_msf_bins', type=int, default=32)
    # masking
    p.add_argument('--mask_ratio', type=float, default=0.0, help='Feature masking ratio (0=off, 0.15=typical)')
    p.add_argument('--topo_mask_ratio', type=float, default=0.0, help='Topology masking ratio (0=off, 0.1=typical)')
    # loss
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.5)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--delta', type=float, default=0.1, help='Feature reconstruction loss weight')
    p.add_argument('--zeta', type=float, default=0.1, help='Topology reconstruction loss weight')
    p.add_argument('--label_smoothing', type=float, default=0.1)
    # training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--pin_memory', action='store_true', default=True)
    p.add_argument('--phase1_epochs', type=int, default=20)
    p.add_argument('--phase2_epochs', type=int, default=30)
    p.add_argument('--phase3_epochs', type=int, default=50)
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--log_level', type=str, default='INFO')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()
def main():
    args = parse_args()
    from src.utils import setup_logging; setup_logging(args.log_level)
    logger = logging.getLogger('META')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True); os.makedirs(args.cache_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f: json.dump(vars(args), f, indent=2)
    # ── Data ──
    from src.processing import process_pdb_directory, create_splits, get_dataloader
    cache_paths = process_pdb_directory(args.data_dir, args.cache_dir, compute_dynamics=args.compute_dynamics,
        n_workers=args.n_workers_process, edge_cutoff=args.edge_cutoff, n_modes=args.n_modes,
        n_conformers=args.n_conformers, per_chain=args.per_chain, min_len=args.min_len, max_len=args.max_len)
    if not cache_paths: logger.error("No data."); sys.exit(1)
    splits = create_splits(cache_paths, args.output_dir, args.val_frac, args.test_frac, args.seed, args.split_file)
    logger.info(f"Data: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    train_dl = get_dataloader(splits['train'], args.batch_size, True, args.num_workers, args.max_len, args.pin_memory)
    val_dl = get_dataloader(splits['val'], args.batch_size, False, args.num_workers, args.max_len, args.pin_memory)
    # ── Auto-detect feature dims ──
    from src.utils import load_features as _lf
    sd = _lf(splits['train'][0])
    d_node = sd['node_feat'].shape[1]; d_edge = sd['edge_feat'].shape[1]
    d_bend = sd['bend_feat'].shape[1] if sd['bend_feat'].shape[0] > 0 else 1
    d_torsion = sd['torsion_feat'].shape[1] if sd['torsion_feat'].shape[0] > 0 else 2
    logger.info(f"Feature dims: d_node={d_node}, d_edge={d_edge}, d_bend={d_bend}, d_torsion={d_torsion}")
    # ── Model ──
    from src.model_utils import METAModel, METALoss
    model = METAModel(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        d_node=d_node, d_edge=d_edge, d_bend=d_bend, d_torsion=d_torsion,
        use_ar=args.use_ar, n_msf_bins=args.n_msf_bins,
        layer_types=args.layer_types, mask_ratio=args.mask_ratio, topo_mask_ratio=args.topo_mask_ratio,
        use_pointer=args.use_pointer, chunk_size=args.chunk_size).to(device)
    from train_utils import count_parameters; logger.info(f"Params: {count_parameters(model):,}, layers: {args.layer_types}, mask={args.mask_ratio}, topo_mask={args.topo_mask_ratio}, pointer={args.use_pointer}")
    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    from train_utils import WarmupCosineScheduler, TrainingCurriculum, EarlyStopping, Timer, to_device, compute_all_metrics, save_checkpoint, load_checkpoint
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)
    curriculum = TrainingCurriculum(args.phase1_epochs, args.phase2_epochs, args.phase3_epochs, args.gamma)
    loss_fn = METALoss(args.alpha, args.beta, 0.0, args.delta, args.zeta, args.label_smoothing)
    early_stop = EarlyStopping(args.patience)
    scaler = torch.amp.GradScaler('cuda') if (args.mixed_precision and device.type == 'cuda') else None
    try:
        from torch.utils.tensorboard import SummaryWriter; writer = SummaryWriter(os.path.join(args.output_dir, 'tb'))
    except ImportError: writer = None
    start_epoch = global_step = 0
    if args.resume:
        meta = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = meta['epoch'] + 1; global_step = meta['step']
    # ── Training ──
    timer = Timer(); best_val = float('inf')
    logger.info(f"Training: {args.epochs} epochs, {len(train_dl)} steps/epoch, {curriculum}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        phase = curriculum.get_phase(epoch); gamma_now = curriculum.get_gamma(epoch)
        use_ar_now = curriculum.use_ar(epoch) and args.use_ar; loss_fn.gamma = gamma_now
        ep_losses = []; t0 = time.time()
        for si, batch in enumerate(train_dl):
            batch = to_device(batch, device); optimizer.zero_grad(set_to_none=True)
            if args.mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(batch); loss, ld = loss_fn(pred, batch, use_ar_now)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(batch); loss, ld = loss_fn(pred, batch, use_ar_now)
                loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
            scheduler.step(); global_step += 1; ep_losses.append(ld)
            if global_step % args.log_every == 0:
                with torch.no_grad(): met = compute_all_metrics(pred, batch, use_ar_now)
                lr = optimizer.param_groups[0]['lr']
                logger.info(f"E{epoch} S{si} | loss={ld['total']:.4f} rec={met.get('recovery',0):.3f} ppl={met.get('perplexity',0):.2f} lr={lr:.2e} P{phase} γ={gamma_now:.3f}")
                if writer:
                    for k, v in ld.items(): writer.add_scalar(f'train/{k}', v, global_step)
                    for k, v in met.items(): writer.add_scalar(f'train/{k}', v, global_step)
        avg_loss = np.mean([d['total'] for d in ep_losses])
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f} time={time.time()-t0:.1f}s")
        # ── Validation ──
        if (epoch+1) % args.eval_every == 0:
            model.eval(); vl = []; vm = []
            with torch.no_grad():
                for batch in val_dl:
                    batch = to_device(batch, device)
                    pred = model(batch); _, ld = loss_fn(pred, batch, use_ar_now)
                    vl.append(ld); vm.append(compute_all_metrics(pred, batch, use_ar_now))
            avl = np.mean([d['total'] for d in vl]); avr = np.mean([m.get('recovery',0) for m in vm])
            logger.info(f"  VAL: loss={avl:.4f} rec={avr:.3f}")
            if writer: writer.add_scalar('val/loss', avl, epoch); writer.add_scalar('val/recovery', avr, epoch)
            if avl < best_val:
                best_val = avl; save_checkpoint(model, optimizer, scheduler, epoch, global_step, {'val_loss': avl}, os.path.join(args.output_dir, 'best_model.pt'), scaler)
            if early_stop.step(avl): logger.info(f"Early stop at epoch {epoch}"); break
        if (epoch+1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, {}, os.path.join(args.output_dir, f'ckpt_e{epoch}.pt'), scaler)
    logger.info(f"Done. Total time: {timer.total():.1f}s")
    save_checkpoint(model, optimizer, scheduler, epoch, global_step, {}, os.path.join(args.output_dir, 'final.pt'), scaler)
    if writer: writer.close()
if __name__ == '__main__': main()
