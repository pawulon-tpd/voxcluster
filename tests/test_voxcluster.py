"""Conformers, envelopes, similarity and clustering."""

import numpy as np
import pytest
from rdkit import Chem
from voxorcist.voxelize import EXC

from voxcluster.analysis import analyse
from voxcluster.conformers import from_smiles, parse_smiles, read_sdf
from voxcluster.occupancy import (
    cluster,
    envelope,
    frame_for,
    heavy_coords,
    occupancy,
    similarity_matrix,
)


def test_parse_smiles_names_and_comments():
    text = "# header\nCCO ethanol\n\nc1ccccc1  # no name\n"
    assert parse_smiles(text) == [("CCO", "ethanol"), ("c1ccccc1", "mol2")]


def test_conformers_differ():
    """Guards against RDKit's seed-0 behaviour, which embeds identical copies."""
    m = from_smiles("CCCCCCCCCC", "decane", n_confs=10)
    assert m.n_confs == 10
    assert np.stack(heavy_coords(m.mol)).std(axis=0).mean() > 0.1


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        from_smiles("not-a-smiles", "bad")
    with pytest.raises(ValueError):
        from_smiles("CCO", "ethanol", seed=0)


def test_flexible_molecule_has_larger_envelope_than_core():
    flexible, _, _ = envelope(from_smiles("CCCCCCCCCCCC(=O)O", "lauric", 20).mol)
    rigid, _, _ = envelope(from_smiles("c1ccc2ccccc2c1", "naphthalene", 20).mol)
    assert flexible["envelope_A3"] > flexible["core_A3"]
    assert rigid["core_fraction"] > flexible["core_fraction"]
    assert rigid["shape"] in {"disc", "rod"} and flexible["shape"] == "rod"


def test_occupancy_is_a_fraction():
    mol = from_smiles("CCCCCC", "hexane", 10).mol
    occ = occupancy(mol, frame_for([mol]))
    assert occ.shape[0] == 8 and occ.min() >= 0 and occ.max() <= 1
    assert occ[EXC].any()


def test_similarity_identity_and_symmetry():
    rng = np.random.default_rng(0)
    occs = [rng.random((8, 4, 4, 4)).astype(np.float32) for _ in range(3)]
    sim = similarity_matrix(occs + [occs[0]], channels=list(range(8)))
    np.testing.assert_allclose(sim, sim.T)
    np.testing.assert_allclose(np.diag(sim), 1.0)
    assert sim[0, 3] == pytest.approx(1.0)
    assert (sim <= 1.0 + 1e-9).all() and (sim >= 0).all()


def test_cluster_cut():
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    labels, order = cluster(sim, min_similarity=0.5)
    assert labels[0] == labels[1] != labels[2]
    assert sorted(order) == [0, 1, 2]


def test_shapes_cluster_apart():
    """Two chains and two fused aromatics should form two clusters."""
    mols = [from_smiles(s, n, 15) for s, n in [
        ("CCCCCCCCCC", "decane"), ("CCCCCCCCCCC", "undecane"),
        ("c1ccc2ccccc2c1", "naphthalene"), ("Cc1ccc2ccccc2c1", "methylnaphthalene")]]
    result = analyse(mols, features="shape", min_similarity=0.7)
    a, b, c, d = result.labels
    assert a == b and c == d and a != c


def test_sdf_records_of_one_molecule_merge(tmp_path):
    m = from_smiles("CCCCO", "butanol", 3)
    path = tmp_path / "poses.sdf"
    with Chem.SDWriter(str(path)) as w:
        for conf in m.mol.GetConformers():
            w.write(m.mol, confId=conf.GetId())
        w.write(from_smiles("c1ccccc1", "benzene", 1).mol)
    molecules = read_sdf(path)
    assert [x.n_confs for x in molecules] == [3, 1]
    assert molecules[0].name == "butanol"
