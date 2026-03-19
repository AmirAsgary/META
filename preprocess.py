"""Standalone preprocessing for META. Processes PDB/CIF to .npz cache, optionally builds LMDB.
Usage: python preprocess.py --pdb_dir ./data/pdbs --cache_dir ./cache
With dynamics: python preprocess.py --pdb_dir ./data/pdbs --cache_dir ./cache --compute_dynamics
Build LMDB:    python preprocess.py --pdb_dir ./data/pdbs --cache_dir ./cache --build_lmdb"""
import argparse, os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logging
from src.processing import process_dataset, create_splits, build_lmdb
def main():
    p = argparse.ArgumentParser(description='META: Preprocess PDB/CIF files')
    p.add_argument('--pdb_dir', type=str, required=True, help='Directory with PDB/CIF files')
    p.add_argument('--cache_dir', type=str, default='./cache', help='Output cache directory for .npz')
    p.add_argument('--output_dir', type=str, default='./output', help='Output dir for splits.json')
    p.add_argument('--compute_dynamics', action='store_true', help='Run ProDy ANM (slow, skip for testing)')
    p.add_argument('--build_lmdb', action='store_true', help='Build LMDB after processing')
    p.add_argument('--lmdb_path', type=str, default=None, help='LMDB output path (default: cache_dir/train.lmdb)')
    p.add_argument('--n_modes', type=int, default=20)
    p.add_argument('--n_conformers', type=int, default=1000)
    p.add_argument('--edge_cutoff', type=float, default=8.0)
    p.add_argument('--n_workers', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--per_chain', action='store_true', default=True)
    p.add_argument('--no_per_chain', dest='per_chain', action='store_false')
    p.add_argument('--min_len', type=int, default=30)
    p.add_argument('--max_len', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_level', type=str, default='INFO')
    args = p.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger('META.preprocess')
    logger.info(f"Processing PDBs from {args.pdb_dir}, dynamics={args.compute_dynamics}")
    cache_paths = process_dataset(args.pdb_dir, args.cache_dir, n_workers=args.n_workers,
        per_chain=args.per_chain, compute_dynamics=args.compute_dynamics, edge_cutoff=args.edge_cutoff,
        n_modes=args.n_modes, n_conformers=args.n_conformers, min_len=args.min_len, max_len=args.max_len)
    if not cache_paths: logger.error("No files processed."); sys.exit(1)
    splits = create_splits(cache_paths, args.output_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    logger.info(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    if args.build_lmdb:
        lp = args.lmdb_path or os.path.join(args.cache_dir, 'train.lmdb')
        build_lmdb(cache_paths, lp)
        logger.info(f"LMDB built: {lp}")
    logger.info("Done.")
if __name__ == '__main__': main()
