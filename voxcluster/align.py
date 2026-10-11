"""Putting conformers into a common frame.

- Own frame: a molecule's conformers superposed on its first conformer by atom
  mapping. Used for "what can this molecule occupy".
- Reference frame: every conformer of every molecule shape-aligned onto one
  conformer of a reference molecule with RDKit's Gaussian shape + colour
  (pharmacophore) overlay. Used to compare molecules with each other.
"""

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdShapeAlign

from .conformers import Molecule


def align_own(m: Molecule) -> Chem.Mol:
    """Copy of the molecule with conformers superposed on conformer 0."""
    mol = Chem.Mol(m.mol)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    rdMolAlign.AlignMolConformers(mol, atomIds=heavy)
    return mol


def align_to_reference(m: Molecule, ref: Molecule, ref_conf: int = 0,
                       shape_weight: float = 0.5) -> tuple[Chem.Mol, list[float]]:
    """Copy of the molecule with each conformer independently shape-aligned
    onto ``ref``'s conformer ``ref_conf``. Returns the aligned copy and the
    per-conformer shape Tanimoto to the reference.

    ``shape_weight`` balances shape (1.0) against colour features (0.0).
    """
    mol = Chem.Mol(m.mol)
    ref_id = ref.mol.GetConformers()[ref_conf].GetId()
    scores = []
    for conf in mol.GetConformers():
        shape, _color = rdShapeAlign.AlignMol(ref.mol, mol, refConfId=ref_id,
                                              probeConfId=conf.GetId(), opt_param=shape_weight)
        scores.append(shape)
    return mol, scores


def default_reference(molecules: list[Molecule]) -> int:
    """Index of the largest molecule (most heavy atoms; first on ties)."""
    return max(range(len(molecules)), key=lambda i: molecules[i].n_heavy)
