"""Dock SMILES into a receptor with AutoDock Vina; poses become conformers.

Receptor and ligands are prepared with Meeko (PDB/SMILES -> PDBQT). Docking
uses the ``vina`` Python package when it imports (Linux wheels on PyPI), else
the official Vina binary via subprocess. Poses come back as RDKit molecules
via Meeko, one :class:`Molecule` per ligand with its poses as conformers, all
in the receptor frame, so they go straight into ``analyse(..., align="none")``.
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .conformers import Molecule

RELEASES = "https://github.com/ccsb-scripps/AutoDock-Vina/releases"


@dataclass(frozen=True)
class Box:
    """Vina search box: centre and edge lengths (A)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def contains(self, xyz: np.ndarray, tol: float = 0.0) -> bool:
        half = np.asarray(self.size) / 2 + tol
        return bool((np.abs(np.asarray(xyz) - np.asarray(self.center)) <= half).all())


def read_reference(path: str | Path) -> Chem.Mol:
    """First molecule of an SDF, MOL, MOL2 or PDB file, hydrogens removed."""
    path = Path(path)
    readers = {".sdf": lambda p: next(iter(Chem.SDMolSupplier(p)), None),
               ".mol": Chem.MolFromMolFile, ".mol2": Chem.MolFromMol2File,
               ".pdb": Chem.MolFromPDBFile}
    if path.suffix.lower() not in readers:
        raise ValueError(f"box ligand must be .sdf, .mol, .mol2 or .pdb, got {path.name}")
    mol = readers[path.suffix.lower()](str(path))
    if mol is None:
        raise ValueError(f"could not read a molecule from {path.name}")
    return Chem.RemoveHs(mol)


def box_from_ligand(path: str | Path, size: float | None = 20.0, padding: float = 5.0) -> Box:
    """Box centred on the reference ligand's heavy-atom centroid: a cube of
    edge ``size``, or with ``size=None`` the ligand extent plus ``padding``
    on every side."""
    xyz = read_reference(path).GetConformer().GetPositions()
    center = tuple(float(c) for c in xyz.mean(axis=0))
    if size is None:
        edges = xyz.max(axis=0) - xyz.min(axis=0) + 2 * padding
        return Box(center, tuple(float(e) for e in edges))
    return Box(center, (float(size),) * 3)


def prepare_receptor(path: str | Path) -> str:
    """Receptor PDBQT text. A ``.pdbqt`` is used as is; a PDB goes through
    Meeko's polymer preparation (residue templates, polar H, rigid receptor).
    Residues Meeko cannot match are dropped."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".pdbqt":
        return text
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates

    prep = MoleculePreparation.from_config({"compute_charges": False})  # template charges
    polymer = Polymer.from_pdb_string(text, ResidueChemTemplates.create_from_defaults(), prep,
                                      allow_bad_res=True)
    rigid, _flex = PDBQTWriterLegacy.write_from_polymer(polymer)
    return rigid


def prepare_ligand(smiles: str, name: str, seed: int = 42) -> str:
    """Ligand PDBQT text: RDKit with H, one ETKDGv3 conformer (MMFF-relaxed)
    as Vina's starting geometry, Meeko typing and torsion tree. Protonation
    is taken from the SMILES as given."""
    if seed <= 0:
        raise ValueError("seed must be positive: RDKit embeds identical conformers with seed 0")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{name}: invalid SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"{name}: RDKit could not embed {smiles!r}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"{name}: Meeko could not write PDBQT: {err.strip()}")
    return pdbqt


def find_vina_binary(path: str | Path | None = None) -> str:
    """Vina executable: ``path``, then ``$VINA_BINARY``, then ``vina`` on
    PATH, then ``~/bin/vina``."""
    if path:
        if Path(path).is_file() and os.access(path, os.X_OK):
            return str(path)
        raise FileNotFoundError(f"Vina binary not found or not executable: {path}")
    for c in (os.environ.get("VINA_BINARY"), shutil.which("vina"), Path.home() / "bin" / "vina"):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    raise FileNotFoundError("AutoDock Vina not found: pip install vina (Linux), or download the "
                            f"binary from {RELEASES} and set VINA_BINARY or pass --vina-binary")


def backend(binary: str | Path | None = None) -> str:
    """``"python"`` if the vina package imports, else the binary's path.
    Raises FileNotFoundError when neither is available."""
    if binary is None:
        try:
            import vina  # noqa: F401
            return "python"
        except ImportError:
            pass
    return find_vina_binary(binary)


def run_vina(receptor: str, ligand: str, box: Box, exhaustiveness: int = 8, n_poses: int = 9,
             seed: int = 42, scoring: str = "vina", binary: str | Path | None = None) -> str:
    """Dock one ligand (PDBQT texts in, output PDBQT text with
    ``REMARK VINA RESULT`` lines out)."""
    engine = backend(binary)
    with tempfile.TemporaryDirectory(prefix="voxcluster-vina-") as tmp:
        rec, lig, out = (Path(tmp) / f for f in ("receptor.pdbqt", "ligand.pdbqt", "out.pdbqt"))
        rec.write_text(receptor)
        if engine == "python":
            from vina import Vina

            v = Vina(sf_name=scoring, seed=seed, verbosity=0)
            v.set_receptor(str(rec))
            v.set_ligand_from_string(ligand)
            v.compute_vina_maps(center=list(box.center), box_size=list(box.size))
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            return v.poses(n_poses=n_poses)
        lig.write_text(ligand)
        cmd = [engine, "--receptor", str(rec), "--ligand", str(lig), "--out", str(out),
               "--scoring", scoring, "--exhaustiveness", str(exhaustiveness),
               "--num_modes", str(n_poses), "--seed", str(seed), "--verbosity", "0"]
        for axis, c, s in zip("xyz", box.center, box.size):
            cmd += [f"--center_{axis}", f"{c:.3f}", f"--size_{axis}", f"{s:.3f}"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"vina failed ({proc.returncode}): {proc.stderr.strip()[-400:]}")
        return out.read_text()


def poses_from_pdbqt(pdbqt: str, name: str) -> Chem.Mol:
    """Vina output -> one RDKit molecule (with H) whose conformers are the
    poses in rank order. Each conformer carries ``vina_score`` (kcal/mol) and
    ``pose_rank`` (1 = best)."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    pdbqt_mol = PDBQTMolecule(pdbqt, skip_typing=True)
    scores = [pose.score for pose in pdbqt_mol]
    mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]
    if mol is None or mol.GetNumConformers() != len(scores):
        raise ValueError(f"{name}: Meeko could not rebuild the poses")
    mol = Chem.Mol(mol)
    for prop in list(mol.GetPropNames()):
        mol.ClearProp(prop)
    for rank, (conf, score) in enumerate(zip(mol.GetConformers(), scores), start=1):
        conf.SetDoubleProp("vina_score", float(score))
        conf.SetIntProp("pose_rank", rank)
    mol.SetProp("_Name", name)
    return mol


def pose_scores(mol: Chem.Mol) -> list[float]:
    """Vina scores of a docked molecule's conformers (NaN where missing)."""
    return [c.GetDoubleProp("vina_score") if c.HasProp("vina_score") else float("nan")
            for c in mol.GetConformers()]


def keep_poses(mol: Chem.Mol, max_delta: float) -> Chem.Mol:
    """Copy keeping only poses within ``max_delta`` kcal/mol of the best."""
    scores = pose_scores(mol)
    best = min(scores)
    out = Chem.Mol(mol)
    for conf, score in zip(mol.GetConformers(), scores):
        if score - best > max_delta:
            out.RemoveConformer(conf.GetId())
    return out


def dock(receptor: str | Path, ligands: list[tuple[str, str]], box: Box,
         exhaustiveness: int = 8, n_poses: int = 9, seed: int = 42, scoring: str = "vina",
         max_delta: float | None = None, binary: str | Path | None = None,
         log: Callable[[str], None] = print) -> tuple[list[Molecule], list[str]]:
    """Dock ``(smiles, name)`` pairs. Returns one Molecule per docked ligand
    (``source="docking"``) and messages for ligands that were skipped; a
    failing ligand never stops the run."""
    backend(binary)  # fail early if Vina is missing
    rec = prepare_receptor(receptor)
    molecules, skipped = [], []
    for i, (smiles, name) in enumerate(ligands, start=1):
        try:
            out = run_vina(rec, prepare_ligand(smiles, name, seed), box, exhaustiveness,
                           n_poses, seed, scoring, binary)
            mol = poses_from_pdbqt(out, name)
            if max_delta is not None:
                mol = keep_poses(mol, max_delta)
        except Exception as err:  # noqa: BLE001 - report and continue with the next ligand
            msg = str(err) if str(err).startswith(name) else f"{name}: {err}"
            skipped.append(msg)
            log(f"[{i}/{len(ligands)}] skipping {msg}")
            continue
        molecules.append(Molecule(name, Chem.MolToSmiles(Chem.RemoveHs(mol)), mol, "docking"))
        log(f"[{i}/{len(ligands)}] {name}: {min(pose_scores(mol)):.2f} kcal/mol, "
            f"{mol.GetNumConformers()} poses")
    return molecules, skipped


def write_poses(molecules: list[Molecule], path: str | Path) -> None:
    """All poses to one SDF, one record per pose with ``vina_score`` and
    ``pose_rank``; ``read_sdf`` groups them back into molecules."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as w:
        for m in molecules:
            for conf in m.mol.GetConformers():
                record = Chem.Mol(m.mol)
                record.SetProp("_Name", m.name)
                if conf.HasProp("vina_score"):
                    record.SetProp("vina_score", f"{conf.GetDoubleProp('vina_score'):.3f}")
                if conf.HasProp("pose_rank"):
                    record.SetIntProp("pose_rank", conf.GetIntProp("pose_rank"))
                w.write(record, confId=conf.GetId())
