"""Molecules in, conformer ensembles out.

SMILES are embedded with RDKit ETKDGv3 and relaxed with MMFF94 (UFF when MMFF
has no parameters). SDF records are used as given and grouped into one
molecule per unique structure, so a docking run's poses become that
compound's conformers.
"""

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class Molecule:
    name: str
    smiles: str
    mol: Chem.Mol  # one or more conformers
    source: str    # "smiles" or "sdf"

    @property
    def n_confs(self) -> int:
        return self.mol.GetNumConformers()

    @property
    def n_heavy(self) -> int:
        return self.mol.GetNumHeavyAtoms()


def parse_smiles(text: str) -> list[tuple[str, str]]:
    """``SMILES [name]`` per line; blank lines and ``#`` comments skipped.
    Unnamed molecules are called mol1, mol2, ..."""
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        smiles, _, name = line.partition(" ")
        out.append((smiles, name.strip() or f"mol{len(out) + 1}"))
    return out


def from_smiles(smiles: str, name: str, n_confs: int = 30, seed: int = 42) -> Molecule:
    """Embed ``n_confs`` conformers. No RMSD pruning: near-duplicates are kept,
    so occupancy reflects how often ETKDG samples each region."""
    if seed <= 0:
        raise ValueError("seed must be positive: RDKit embeds identical conformers with seed 0")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{name}: invalid SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if not AllChem.EmbedMultipleConfs(mol, n_confs, params):
        raise ValueError(f"{name}: RDKit could not embed {smiles!r}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500)
    mol.SetProp("_Name", name)
    return Molecule(name, Chem.MolToSmiles(Chem.RemoveHs(mol)), mol, "smiles")


def read_sdf(path: str | Path) -> list[Molecule]:
    """Group SDF records by canonical SMILES; records of one structure become
    conformers of one molecule, in file order. Hydrogens are dropped so poses
    with and without explicit H (e.g. Vina's polar-only H) merge cleanly."""
    groups: dict[str, Molecule] = {}
    for record in Chem.SDMolSupplier(str(path)):
        if record is None:
            continue
        smiles = Chem.MolToSmiles(record)
        if smiles not in groups:
            name = record.GetProp("_Name").strip() if record.HasProp("_Name") else ""
            groups[smiles] = Molecule(name or smiles, smiles, Chem.Mol(record), "sdf")
            continue
        first = groups[smiles].mol
        match = record.GetSubstructMatch(first)  # record atom for each atom of first
        ordered = Chem.RenumberAtoms(record, list(match))
        first.AddConformer(ordered.GetConformer(), assignId=True)
    return list(groups.values())
