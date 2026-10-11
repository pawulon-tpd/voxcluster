"""End-to-end analysis: envelopes, alignment, similarity, clusters."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem
from voxorcist.voxelize import CHANNELS, EXC

from .align import align_own, align_to_reference, default_reference
from .conformers import Molecule
from .occupancy import Frame, cluster, envelope, frame_for, occupancy, similarity_matrix

CHANNEL_SETS = {"shape": [EXC], "shape+features": list(range(len(CHANNELS)))}


@dataclass
class Analysis:
    molecules: list[Molecule]
    reference: int
    frame: Frame                   # shared comparison box
    aligned: list[Chem.Mol]        # conformers in the shared frame
    occupancies: list[np.ndarray]  # (8, D, H, W) per molecule, shared frame
    similarity: np.ndarray
    labels: np.ndarray
    order: np.ndarray              # dendrogram leaf order
    table: pd.DataFrame
    own: list[tuple[Chem.Mol, np.ndarray, Frame]] = field(default_factory=list)  # envelopes

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.molecules]


def analyse(
    molecules: list[Molecule],
    reference: int | None = None,
    align: str = "shape",
    features: str = "shape+features",
    min_similarity: float = 0.5,
    resolution: float = 1.0,
) -> Analysis:
    """``align="shape"`` overlays every conformer on the reference's first
    conformer; ``align="none"`` keeps input coordinates (poses that already
    share a frame, e.g. docking output)."""
    if not molecules:
        raise ValueError("no molecules")
    ref = default_reference(molecules) if reference is None else reference

    rows, own = [], []
    for m in molecules:
        mol = align_own(m)
        metrics, occ, frame = envelope(mol, resolution)
        own.append((mol, occ, frame))
        rows.append({"name": m.name, **metrics, "smiles": m.smiles})

    if align == "shape":
        pairs = [align_to_reference(m, molecules[ref]) for m in molecules]
        aligned = [mol for mol, _ in pairs]
        scores = [float(np.mean(s)) for _, s in pairs]
    elif align == "none":
        aligned, scores = [Chem.Mol(m.mol) for m in molecules], [float("nan")] * len(molecules)
    else:
        raise ValueError(f"align must be 'shape' or 'none', got {align!r}")

    frame = frame_for(aligned, resolution)
    occs = [occupancy(mol, frame) for mol in aligned]
    sim = similarity_matrix(occs, CHANNEL_SETS[features])
    labels, order = cluster(sim, min_similarity)

    table = pd.DataFrame(rows)
    table.insert(1, "cluster", labels)
    table.insert(2, "shape_tanimoto_to_ref", scores)
    return Analysis(molecules, ref, frame, aligned, occs, sim, labels, order, table, own)
