"""Occupancy grids, envelope metrics, similarity and clustering.

A conformer occupies a voxel when its density there reaches exp(-2), the
single-atom density at the van der Waals radius (voxorcist's convention). A
molecule's occupancy is the fraction of its conformers occupying each voxel,
per pharmacophore channel.
"""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from voxorcist.poses import SURFACE_LEVEL
from voxorcist.voxelize import CHANNELS, EXC, Atoms, GridSpec, grid_torch, ligand_features

SHAPE_VERTICES = {"rod": (0.0, 1.0), "disc": (0.5, 0.5), "sphere": (1.0, 1.0)}


@dataclass(frozen=True)
class Frame:
    """A grid box: centre plus geometry."""

    center: np.ndarray
    spec: GridSpec

    @property
    def voxel_volume(self) -> float:
        return self.spec.resolution**3


def heavy_coords(mol: Chem.Mol) -> list[np.ndarray]:
    """Heavy-atom coordinates of every conformer, in ``ligand_features`` order."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return [c.GetPositions()[heavy].astype(np.float32) for c in mol.GetConformers()]


def frame_for(mols: list[Chem.Mol], resolution: float = 1.0, margin: float = 3.0) -> Frame:
    """Smallest cubic box holding every conformer of every molecule, plus a
    margin for the density tail (1.5 vdW radii is about 3 A)."""
    xyz = np.concatenate([c for m in mols for c in heavy_coords(m)])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    size = float(np.ceil(((hi - lo).max() + 2 * margin) / resolution) * resolution)
    return Frame((lo + hi) / 2, GridSpec(resolution=resolution, size=size))


def occupancy(mol: Chem.Mol, frame: Frame) -> np.ndarray:
    """Fraction of conformers occupying each voxel: ``(8, D, H, W)``."""
    _, types, radii = ligand_features(mol)  # typing depends on topology only
    center = np.asarray(frame.center, np.float32)
    hits = np.zeros((len(CHANNELS),) + (frame.spec.dim,) * 3, np.float32)
    confs = heavy_coords(mol)
    for coords in confs:
        hits += grid_torch(Atoms(coords, types, radii, center), frame.spec) >= SURFACE_LEVEL
    return hits / len(confs)


def shape_class(npr1: float, npr2: float) -> str:
    """Nearest corner of the normalised-PMI triangle."""
    return min(SHAPE_VERTICES, key=lambda k: np.hypot(npr1 - SHAPE_VERTICES[k][0],
                                                      npr2 - SHAPE_VERTICES[k][1]))


def envelope(mol: Chem.Mol, resolution: float = 1.0) -> tuple[dict, np.ndarray, Frame]:
    """What a molecule can occupy, from its own superposed conformers.

    Returns metrics, the occupancy grid and its frame. ``envelope_A3`` is the
    volume reached by any conformer, ``core_A3`` the volume occupied by at
    least half of them; their ratio is a flexibility measure.
    """
    frame = frame_for([mol], resolution)
    occ = occupancy(mol, frame)
    v = frame.voxel_volume
    ids = [c.GetId() for c in mol.GetConformers()]
    npr1 = float(np.mean([rdMolDescriptors.CalcNPR1(mol, confId=i) for i in ids]))
    npr2 = float(np.mean([rdMolDescriptors.CalcNPR2(mol, confId=i) for i in ids]))
    metrics = {
        "n_confs": len(ids),
        "envelope_A3": float((occ[EXC] > 0).sum() * v),
        "core_A3": float((occ[EXC] >= 0.5).sum() * v),
        "npr1": npr1,
        "npr2": npr2,
        "shape": shape_class(npr1, npr2),
    }
    metrics["core_fraction"] = metrics["core_A3"] / metrics["envelope_A3"]
    for c, name in enumerate(CHANNELS[:6]):  # pharmacophore channels with a ligand meaning
        metrics[f"{name}_A3"] = float((occ[c] > 0).sum() * v)
    return metrics, occ, frame


def similarity_matrix(occs: list[np.ndarray], channels: list[int]) -> np.ndarray:
    """Weighted (Ruzicka) Tanimoto between occupancy grids:
    sum(min(a, b)) / sum(max(a, b)) over the selected channels."""
    x = np.stack([o[channels].reshape(-1) for o in occs])
    n = len(x)
    sim = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            hi = np.maximum(x[i], x[j]).sum()
            sim[i, j] = sim[j, i] = np.minimum(x[i], x[j]).sum() / hi if hi else 0.0
    return sim


def cluster(sim: np.ndarray, min_similarity: float) -> tuple[np.ndarray, np.ndarray]:
    """Average-linkage clusters joined while their mean similarity is at least
    ``min_similarity``. Returns 1-based labels and the dendrogram leaf order."""
    n = len(sim)
    if n == 1:
        return np.array([1]), np.array([0])
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method="average")
    return fcluster(tree, t=1.0 - min_similarity, criterion="distance"), leaves_list(tree)
