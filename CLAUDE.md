# voxcluster — project context

Cluster molecules by voxelised 3D occupancy. Input: SMILES (conformers via
RDKit ETKDG) or SDF poses. Output: per-molecule envelopes, a similarity
matrix, clusters, and a Gradio app with 3D voxel views.

## Decisions

- Typing and gridding come from the `voxorcist` package; do not copy them here.
- Two frames, never mixed: own frame (conformers superposed by atom mapping)
  for envelope metrics; shared frame (shape + colour alignment onto one
  reference conformer, or input coordinates with `align="none"`) for comparison.
- Occupancy threshold exp(-2); similarity is weighted Tanimoto; clustering is
  average linkage on 1 - similarity.
- ETKDG seed must be > 0: RDKit 2026.09 embeds identical conformers with seed 0.
  `tests/test_voxcluster.py::test_conformers_differ` guards this.
- Docking (`docking.py`, `python -m voxcluster.dock`) is optional: Meeko prepares
  receptor/ligands, Vina runs via the `vina` package if it imports (Linux x86_64
  wheels) else the official binary (`--vina-binary`, `$VINA_BINARY`, PATH, `~/bin/vina`).
  Docked poses share the receptor frame, so they are analysed with `align="none"`.
- Never commit Vina binaries or PDBQT files; `tests/test_docking.py` skips without Vina.

## Conventions

- Python >= 3.10; ruff, line length 100; tests with pytest.
- The app lives in `app.py` at the root (Hugging Face Space layout).
- Keep it small: no database, no job queue, no Docker.
