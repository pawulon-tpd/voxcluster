"""``python -m voxcluster.dock receptor.pdb ligands.smi --box-ligand ref.sdf``.

Docks each SMILES with AutoDock Vina and writes all poses to one SDF; cluster
them with ``python -m voxcluster <out.sdf> --align none``.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from .conformers import parse_smiles
from .docking import Box, backend, box_from_ligand, dock, pose_scores, write_poses


def main() -> None:
    p = argparse.ArgumentParser(description="Dock SMILES into a receptor with AutoDock Vina.")
    p.add_argument("receptor", help="receptor PDB (prepared with Meeko) or PDBQT")
    p.add_argument("ligands", help="SMILES file ('SMILES name' per line)")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--box-ligand", help="reference ligand (SDF/MOL/MOL2/PDB); box centre "
                       "= its heavy-atom centroid")
    where.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--size", default="20", help="cubic box edge in A, or 'auto' (box ligand "
                   "extent + --padding)")
    p.add_argument("--padding", type=float, default=5.0, help="for --size auto")
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--n-poses", type=int, default=9)
    p.add_argument("--max-delta", type=float, default=None,
                   help="keep only poses within this many kcal/mol of each ligand's best")
    p.add_argument("--scoring", choices=["vina", "vinardo"], default="vina")
    p.add_argument("--seed", type=int, default=42, help="RDKit and Vina seed, must be > 0")
    p.add_argument("--vina-binary", default=None, help="Vina executable (default: python "
                   "package, $VINA_BINARY, PATH, ~/bin/vina)")
    p.add_argument("--out", default="results/docked.sdf")
    args = p.parse_args()

    if args.size == "auto":
        if not args.box_ligand:
            raise SystemExit("--size auto needs --box-ligand")
        box = box_from_ligand(args.box_ligand, None, args.padding)
    elif args.box_ligand:
        box = box_from_ligand(args.box_ligand, float(args.size))
    else:
        box = Box(tuple(args.center), (float(args.size),) * 3)
    try:
        engine = backend(args.vina_binary)
    except FileNotFoundError as err:
        raise SystemExit(str(err))
    print(f"vina: {engine}   box centre {', '.join(f'{c:.2f}' for c in box.center)}   "
          f"size {' x '.join(f'{s:.1f}' for s in box.size)} A")

    start = time.perf_counter()
    molecules, skipped = dock(args.receptor, parse_smiles(Path(args.ligands).read_text()), box,
                              args.exhaustiveness, args.n_poses, args.seed, args.scoring,
                              args.max_delta, args.vina_binary)
    elapsed = time.perf_counter() - start
    if not molecules:
        raise SystemExit("no ligand docked")
    write_poses(molecules, args.out)

    table = pd.DataFrame([{"name": m.name, "best_kcal_mol": min(pose_scores(m.mol)),
                           "n_poses": m.n_confs} for m in molecules])
    print(table.to_string(index=False, float_format="%.2f"))
    if skipped:
        print(f"skipped {len(skipped)}: " + "; ".join(skipped))
    n = len(molecules) + len(skipped)
    print(f"{elapsed:.1f} s ({elapsed / n:.1f} s per ligand) -> {args.out}")
    print(f"cluster with: python -m voxcluster {args.out} --align none")


if __name__ == "__main__":
    main()
