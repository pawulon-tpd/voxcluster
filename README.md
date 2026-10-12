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

## Docking with Vina

Give a receptor, a box and SMILES; voxcluster docks each ligand with
[AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina), keeps the
poses, and clusters them in the receptor frame (`--align none`), so clusters
reflect how ligands occupy the actual binding site.

```bash
pip install -e ".[docking]"   # meeko + gemmi; the vina package on Linux x86_64
python -m voxcluster.dock examples/3ptb/pocket.pdb examples/trypsin_ligands.smi \
    --box-ligand examples/3ptb/ligand.sdf --out results/docked.sdf
python -m voxcluster results/docked.sdf --align none
```

- **Receptor:** PDB prepared with [Meeko](https://github.com/forlilab/Meeko)
  (residue templates, polar hydrogens, rigid receptor; unmatched residues are
  dropped), or a ready PDBQT.
- **Box:** centred on a reference ligand's heavy-atom centroid
  (`--box-ligand`, SDF/MOL/MOL2/PDB) or `--center X Y Z`; 20 Å cube by
  default (`--size`), or `--size auto` = ligand extent + `--padding` (5 Å).
- **Ligands:** SMILES → RDKit with H → one ETKDGv3 conformer (seed 42) →
  Meeko PDBQT → Vina (`vina` scoring, `--exhaustiveness 8`, `--n-poses 9`,
  fixed `--seed`). Poses come back via Meeko with `vina_score` (kcal/mol) and
  `pose_rank` as SD properties in one SDF. By default only poses within
  1 kcal/mol of each ligand's best are kept (`--max-delta`, e.g. 99 keeps all).
  Ligands that fail preparation or docking are skipped with a message.
- **App:** open "Dock into a receptor" under the inputs, upload a receptor and
  a box ligand (or click "Load trypsin example"); the table gains
  `vina_best`, and the shared-frame 3D view draws the pocket as grey sticks.

**Vina itself.** The `vina` Python package is used when it imports; PyPI only
has wheels for it on Linux x86_64 (it does not build on macOS arm64 without
Boost). Otherwise voxcluster calls the official binary from the
[releases page](https://github.com/ccsb-scripps/AutoDock-Vina/releases)
(Apache-2.0), found via `--vina-binary`, then `$VINA_BINARY`, then `vina` on
`PATH`, then `~/bin/vina`.

**Redocking check.** Benzamidine from SMILES into the trypsin pocket of PDB
3PTB (box on the crystal ligand, exhaustiveness 8): top pose 0.39–0.42 Å
heavy-atom RMSD from the crystal pose (symmetry-aware, no realignment) over
seeds 7, 42 and 123, best score −6.1 kcal/mol, about 0.7 s per ligand on an
Apple M-series laptop with this small (10 Å) pocket. Larger receptors and
boxes, and flexible ligands, take longer: seconds to a minute per ligand.

**Limitations.**
- Rigid receptor; waters and cofactors are whatever the PDB contains.
- Protonation is taken from the SMILES as given: no protonation-state or
  tautomer enumeration (benzamidines are docked as neutral amidines).
- Vina scores rank poses; they are not binding affinities.
- The box decides what can be found: centre it on the site of interest and
  make it large enough for the biggest ligand.
- Clusters use all kept poses, weighted equally. The 1 kcal/mol default
  drops low-ranked poses that would widen the occupancy; the cut-off still
  affects the clusters (with all 9 poses the trypsin example splits differently).

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

MIT, see [LICENSE](LICENSE). Optional docking dependencies are used as
external programs or libraries under their own licences: AutoDock Vina
(Apache-2.0), Meeko (LGPL-2.1), gemmi (MPL-2.0). The 3PTB example files come
from the RCSB PDB (public domain).
