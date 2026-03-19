"""Standalone data preprocessing script for META.
Usage: python preprocess.py --pdb_dir ./data/pdbs --cache_dir ./cache --compute_dynamics
"""
import argparse, os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logging
from src.processing import process_pdb_directory, create_splits
def main():
    p = argparse.ArgumentParser(description='META: Preprocess PDB files')
    p.add_argument('--pdb_dir', type=str, required=True, help='Directory with PDB files')
    p.add_argument('--cache_dir', type=str, default='./cache', help='Output cache directory')
    p.add_argument('--output_dir', type=str, default='./output', help='Output dir for splits')
    p.add_argument('--compute_dynamics', action='store_true', help='Run ProDy ANM')
    p.add_argument('--n_modes', type=int, default=20)
    p.add_argument('--n_conformers', type=int, default=1000)
    p.add_argument('--edge_cutoff', type=float, default=8.0)
    p.add_argument('--n_workers', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--per_chain', action='store_true', default=True, help='Process each chain separately')
    p.add_argument('--no_per_chain', dest='per_chain', action='store_false', help='Process as complex')
    p.add_argument('--min_len', type=int, default=30)
    p.add_argument('--max_len', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_level', type=str, default='INFO')
    args = p.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger('META.preprocess')
    logger.info("Starting preprocessing...")
    cache_paths = process_pdb_directory(
        args.pdb_dir, args.cache_dir, compute_dynamics=args.compute_dynamics,
        n_workers=args.n_workers, edge_cutoff=args.edge_cutoff,
        n_modes=args.n_modes, n_conformers=args.n_conformers,
        per_chain=args.per_chain, min_len=args.min_len, max_len=args.max_len)
    if not cache_paths:
        logger.error("No files processed."); sys.exit(1)
    splits = create_splits(cache_paths, args.output_dir,
                           val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    logger.info("Done. Train=%d, Val=%d, Test=%d" % (len(splits['train']), len(splits['val']), len(splits['test'])))
if __name__ == '__main__':
    main()
