"""``python -m voxcluster molecules.smi`` (or ``poses.sdf``)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import CHANNEL_SETS, analyse
from .conformers import from_smiles, parse_smiles, read_sdf


def main() -> None:
    p = argparse.ArgumentParser(description="Cluster molecules by voxel occupancy.")
    p.add_argument("input", help="SMILES file ('SMILES name' per line) or SDF of 3D poses")
    p.add_argument("--n-confs", type=int, default=30, help="conformers per SMILES")
    p.add_argument("--reference", default=None, help="molecule name for the shared frame "
                   "(default: largest molecule)")
    p.add_argument("--align", choices=["shape", "none"], default="shape",
                   help="'none' keeps SDF coordinates, e.g. poses docked into one pocket")
    p.add_argument("--features", choices=sorted(CHANNEL_SETS), default="shape+features")
    p.add_argument("--min-similarity", type=float, default=0.5, help="cluster cut-off")
    p.add_argument("--seed", type=int, default=42, help="ETKDG seed, must be > 0")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    path = Path(args.input)
    if path.suffix == ".sdf":
        molecules = read_sdf(path)
    else:
        molecules = []
        for smiles, name in parse_smiles(path.read_text()):
            try:
                molecules.append(from_smiles(smiles, name, args.n_confs, args.seed))
            except ValueError as err:
                print(f"skipping: {err}")
    names = [m.name for m in molecules]
    if args.reference is not None and args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} not among {names}")
    ref = names.index(args.reference) if args.reference else None

    result = analyse(molecules, ref, args.align, args.features, args.min_similarity)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "molecules.csv", index=False)
    pd.DataFrame(result.similarity, index=names, columns=names).to_csv(out / "similarity.csv")
    np.savez_compressed(out / "occupancy.npz", names=np.array(names),
                        occupancy=np.stack(result.occupancies), center=result.frame.center,
                        resolution=result.frame.spec.resolution)

    cols = ["name", "cluster", "n_confs", "envelope_A3", "core_A3", "shape", "shape_tanimoto_to_ref"]
    print(f"reference: {names[result.reference]}   box: {result.frame.spec.size:g} A")
    print(result.table[cols].sort_values(["cluster", "name"]).to_string(index=False, float_format="%.2f"))
    print(f"-> {out}/ (molecules.csv, similarity.csv, occupancy.npz)")


if __name__ == "__main__":
    main()
