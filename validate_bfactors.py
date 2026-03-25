"""validate_bfactors.py — Fix #5: B-factor validation per proposal §2.1.4.
Computes Pearson correlation between ANM-predicted MSF and experimental
B-factors for each protein. Flags proteins with rho < 0.5.
Usage: python validate_bfactors.py --cache_dir ./cache --pdb_dir ./pdbs --output flagged.json
"""
import argparse, json, logging, os, sys, glob
import numpy as np
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
# ── Extract experimental B-factors from PDB/CIF ──
def extract_bfactors_pdb(pdb_path, chain_id=None):
    """Extract per-residue mean isotropic B-factor for CA atoms from PDB file."""
    bfacs, res_seen = {}, set()
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')): continue
            aname = line[12:16].strip()
            if aname != 'CA': continue
            ch = line[21].strip()
            if chain_id and ch != chain_id: continue
            resnum = int(line[22:26].strip())
            icode = line[26].strip()
            key = (ch, resnum, icode)
            if key in res_seen: continue
            res_seen.add(key)
            try: bf = float(line[60:66].strip())
            except ValueError: continue
            bfacs[key] = bf
    if not bfacs: return None
    keys_sorted = sorted(bfacs.keys(), key=lambda x: (x[0], x[1], x[2]))
    return np.array([bfacs[k] for k in keys_sorted], dtype=np.float32)
def extract_bfactors_cif(cif_path, chain_id=None):
    """Extract per-residue mean B-factor for CA atoms from mmCIF file."""
    try: from Bio.PDB import MMCIFParser
    except ImportError:
        logger.warning("BioPython required for mmCIF B-factor extraction"); return None
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure('s', cif_path)
    bfacs = []
    for model in struct:
        for chain in model:
            if chain_id and chain.id != chain_id: continue
            for res in chain.get_residues():
                if res.id[0] != ' ': continue  # skip hetero
                for atom in res:
                    if atom.name == 'CA':
                        bfacs.append(atom.bfactor); break
        break  # first model only
    return np.array(bfacs, dtype=np.float32) if bfacs else None
def extract_bfactors(path, chain_id=None):
    """Auto-detect PDB vs CIF and extract B-factors."""
    ext = Path(path).suffix.lower()
    if ext in ('.cif', '.mmcif'): return extract_bfactors_cif(path, chain_id)
    return extract_bfactors_pdb(path, chain_id)
# ── Convert B-factor to MSF: B = (8*pi^2/3) * <dr^2> ──
def bfactor_to_msf(bfactors):
    """Convert isotropic B-factors to mean-square fluctuation."""
    return bfactors * 3.0 / (8.0 * np.pi ** 2)
# ── Pearson correlation ──
def pearson_r(x, y):
    """Compute Pearson correlation, return NaN if degenerate."""
    if len(x) < 3 or x.std() < 1e-10 or y.std() < 1e-10: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])
# ── Main validation loop ──
def validate_dataset(cache_dir, pdb_dir, rho_threshold=0.5, chain_map=None):
    """Validate ANM MSF against experimental B-factors for all cached proteins.
    Returns dict of {name: {rho, n_res, flagged, ...}} for every protein with B-factor data."""
    dyn_files = sorted(glob.glob(os.path.join(cache_dir, '*_dyn.npz')))
    if not dyn_files:
        logger.error(f"No _dyn.npz files found in {cache_dir}"); return {}
    results = {}
    n_ok, n_flagged, n_skip = 0, 0, 0
    for dpath in dyn_files:
        name = Path(dpath).stem.replace('_dyn', '')
        # load ANM MSF
        dyn = np.load(dpath, allow_pickle=True)
        if 'msf' not in dyn:
            logger.debug(f"  {name}: no MSF in dynamics file, skipping"); n_skip += 1; continue
        anm_msf = dyn['msf']
        # find corresponding PDB/CIF
        chain_id = None
        if chain_map and name in chain_map: chain_id = chain_map[name]
        # try to find PDB file
        pdb_path = None
        for ext in ['.pdb', '.cif', '.pdb.gz', '.cif.gz']:
            # strip chain suffix if present (e.g., "1abc_A" -> "1abc")
            base = name.split('_')[0] if '_' in name else name
            cand = os.path.join(pdb_dir, base + ext)
            if os.path.exists(cand): pdb_path = cand; break
            cand = os.path.join(pdb_dir, name + ext)
            if os.path.exists(cand): pdb_path = cand; break
        if pdb_path is None:
            logger.debug(f"  {name}: no PDB/CIF found, skipping B-factor validation"); n_skip += 1; continue
        # extract chain_id from name if format is "pdbid_chain"
        if chain_id is None and '_' in name:
            parts = name.split('_')
            if len(parts) >= 2 and len(parts[-1]) == 1: chain_id = parts[-1]
        # handle gzipped files
        actual_path = pdb_path
        if pdb_path.endswith('.gz'):
            import gzip, tempfile
            with gzip.open(pdb_path, 'rt') as gz:
                tmp = tempfile.NamedTemporaryFile(suffix=Path(pdb_path).suffixes[0], delete=False, mode='w')
                tmp.write(gz.read()); tmp.close(); actual_path = tmp.name
        bfacs = extract_bfactors(actual_path, chain_id)
        if actual_path != pdb_path: os.unlink(actual_path)  # cleanup temp
        if bfacs is None or len(bfacs) == 0:
            logger.debug(f"  {name}: no B-factors extracted"); n_skip += 1; continue
        # length check
        if len(bfacs) != len(anm_msf):
            logger.warning(f"  {name}: length mismatch B-fac={len(bfacs)} vs ANM={len(anm_msf)}, skipping")
            n_skip += 1; continue
        # convert B-factors to MSF
        exp_msf = bfactor_to_msf(bfacs)
        rho = pearson_r(anm_msf, exp_msf)
        flagged = np.isnan(rho) or rho < rho_threshold
        results[name] = {'rho': round(rho, 4) if not np.isnan(rho) else None,
                         'n_res': int(len(anm_msf)), 'flagged': flagged,
                         'mean_bfac': round(float(bfacs.mean()), 2),
                         'mean_anm_msf': round(float(anm_msf.mean()), 6)}
        if flagged: n_flagged += 1; logger.info(f"  {name}: FLAGGED rho={rho:.3f}")
        else: n_ok += 1
    logger.info(f"Validation complete: {n_ok} OK, {n_flagged} flagged (rho<{rho_threshold}), {n_skip} skipped")
    return results
# ── Integration with preprocessing: add to process_single_structure ──
def validate_single_protein(backbone, dynamics, pdb_path, chain_id=None, rho_threshold=0.5):
    """Validate a single protein during preprocessing. Returns (rho, flagged) or (None, False).
    Call this inside process_single_structure() after compute_anm_dynamics()."""
    if dynamics is None: return None, False
    bfacs = extract_bfactors(pdb_path, chain_id)
    if bfacs is None or len(bfacs) != len(dynamics['msf']): return None, False
    exp_msf = bfactor_to_msf(bfacs)
    rho = pearson_r(dynamics['msf'], exp_msf)
    return (round(rho, 4) if not np.isnan(rho) else None), (np.isnan(rho) or rho < rho_threshold)
# ── CLI ──
def main():
    p = argparse.ArgumentParser(description='Validate ANM MSF against experimental B-factors')
    p.add_argument('--cache_dir', required=True, help='Directory with _dyn.npz files')
    p.add_argument('--pdb_dir', required=True, help='Directory with PDB/CIF files')
    p.add_argument('--output', default='bfactor_validation.json', help='Output JSON file')
    p.add_argument('--rho_threshold', type=float, default=0.5, help='Pearson rho threshold for flagging')
    p.add_argument('--flagged_list', default=None, help='Output text file listing flagged protein names (one per line)')
    args = p.parse_args()
    results = validate_dataset(args.cache_dir, args.pdb_dir, args.rho_threshold)
    with open(args.output, 'w') as f: json.dump(results, f, indent=2)
    logger.info(f"Results written to {args.output}")
    if args.flagged_list:
        flagged = [k for k, v in results.items() if v['flagged']]
        with open(args.flagged_list, 'w') as f: f.write('\n'.join(flagged) + '\n')
        logger.info(f"Flagged list ({len(flagged)} proteins) written to {args.flagged_list}")
    # summary stats
    rhos = [v['rho'] for v in results.values() if v['rho'] is not None]
    if rhos:
        logger.info(f"Pearson rho stats: mean={np.mean(rhos):.3f}, median={np.median(rhos):.3f}, "
                     f"min={np.min(rhos):.3f}, max={np.max(rhos):.3f}")
if __name__ == '__main__': main()
