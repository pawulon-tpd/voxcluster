---
title: voxcluster
emoji: 🧊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
license: mit
---

# voxcluster

**Cluster molecules by the space they occupy.** Give it SMILES (or 3D poses
in an SDF); it generates conformers, aligns everything into one frame,
voxelises each molecule's conformer ensemble into pharmacophore channels, and
clusters molecules by how much their occupancy overlaps.

Built on [voxorcist](https://github.com/pawulon-tpd/voxorcist) for atom typing
and gridding.

## Quickstart

```bash
pip install -e ".[app]"                  # pulls voxorcist from GitHub
python -m voxcluster examples/molecules.smi
python app.py                            # browser UI at http://127.0.0.1:7860
```

The CLI prints a table of molecules with cluster labels and writes
`results/molecules.csv`, `results/similarity.csv` and `results/occupancy.npz`.

## How it works

1. **Conformers.** SMILES → RDKit ETKDGv3, 30 conformers by default, relaxed
   with MMFF94 (UFF fallback). SDF records are grouped by structure, so all
   poses of one compound (e.g. a docking run) become its conformers.
2. **What a molecule can occupy (envelope).** Each molecule's conformers are
   superposed on its first conformer and voxelised (1 Å grid). Reported per
   molecule:
   - `envelope_A3`: volume reached by any conformer
   - `core_A3`: volume occupied by at least half of them
   - `core_fraction`: core / envelope, a flexibility measure (1.0 = rigid)
   - per-channel envelope volumes (hydrophobic, aromatic, donor, acceptor,
     positive, negative)
   - `shape`: rod / disc / sphere from normalised principal moments (NPR)
3. **Shared frame.** Every conformer of every molecule is aligned onto the
   first conformer of a reference molecule (default: the largest; choose any)
   with RDKit's Gaussian shape + pharmacophore-colour overlay (`rdShapeAlign`).
   For poses already in a common frame (e.g. docked into one pocket), use
   `--align none` to keep the coordinates.
4. **Occupancy and similarity.** In the shared box, a molecule's occupancy is
   the fraction of its conformers occupying each voxel (density ≥ exp(−2), the
   single-atom value at the van der Waals radius). Similarity is the weighted
   Tanimoto Σmin / Σmax over the 8 channels (`shape+features`) or over volume
   only (`shape`).
5. **Clusters.** Average-linkage hierarchical clustering on 1 − similarity,
   cut at a minimum similarity (default 0.5; for `shape` alone use about 0.7,
   since shape-aligned molecules always overlap substantially).

## Limitations

- Without a protein, the shared frame comes from shape alignment, so clusters
  reflect 3D shape and feature layout, not binding. This is the same idea as
  ROCS-style shape similarity; the voxel channels and envelopes are the
  additions.
- Results depend on the reference molecule and on conformer sampling;
  conformers are not energy-filtered or Boltzmann-weighted.
- Each conformer is aligned to the reference independently, so a flexible
  molecule's occupancy is the union of its best overlays, not of one rigid
  placement.
- Practical size: tens to hundreds of molecules (about 0.3 s per molecule at
  30 conformers on a laptop); the app caps input at 60.

## Licence

MIT, see [LICENSE](LICENSE).
