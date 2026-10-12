"""Vina docking: skipped unless meeko and Vina (package or binary) are available."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolAlign

pytest.importorskip("meeko")
pytest.importorskip("gemmi")  # meeko imports it without declaring it

from voxcluster.analysis import analyse
from voxcluster.conformers import read_sdf
from voxcluster.docking import (
    Box,
    backend,
    box_from_ligand,
    dock,
    find_vina_binary,
    keep_poses,
    pose_scores,
    read_reference,
    write_poses,
)

try:
    backend()
except FileNotFoundError:
    pytest.skip("neither the vina package nor a Vina binary is available", allow_module_level=True)

EXAMPLE = Path(__file__).parent.parent / "examples" / "3ptb"
POCKET, CRYSTAL = EXAMPLE / "pocket.pdb", EXAMPLE / "ligand.sdf"
LIGANDS = [("NC(=N)c1ccccc1", "benzamidine"), ("NCc1ccccc1", "benzylamine")]


@pytest.fixture(scope="module")
def docked():
    box = box_from_ligand(CRYSTAL)
    molecules, skipped = dock(POCKET, LIGANDS + [("not-a-smiles", "bad")], box,
                              exhaustiveness=2, n_poses=3, log=lambda msg: None)
    return box, molecules, skipped


def test_box_from_ligand():
    xyz = read_reference(CRYSTAL).GetConformer().GetPositions()
    cube = box_from_ligand(CRYSTAL, size=16)
    np.testing.assert_allclose(cube.center, xyz.mean(axis=0))
    assert cube.size == (16.0, 16.0, 16.0)
    auto = box_from_ligand(CRYSTAL, size=None, padding=4)
    np.testing.assert_allclose(auto.size, xyz.max(axis=0) - xyz.min(axis=0) + 8)


def test_bad_ligand_is_skipped_not_fatal(docked):
    _, molecules, skipped = docked
    assert [m.name for m in molecules] == ["benzamidine", "benzylamine"]
    assert len(skipped) == 1 and skipped[0].startswith("bad")


def test_poses_inside_box_with_negative_scores(docked):
    box, molecules, _ = docked
    for m in molecules:
        assert m.source == "docking" and 1 <= m.n_confs <= 3
        scores = pose_scores(m.mol)
        assert all(isinstance(s, float) and s < 0 for s in scores)
        assert scores == sorted(scores)
        for conf in m.mol.GetConformers():
            assert box.contains(conf.GetPositions(), tol=0.5)


def test_redocked_benzamidine_matches_crystal(docked):
    _, molecules, _ = docked
    pose = Chem.RemoveHs(molecules[0].mol)
    first = pose.GetConformers()[0].GetId()
    assert rdMolAlign.CalcRMS(pose, read_reference(CRYSTAL), prbId=first) < 2.0


def test_sdf_round_trip_groups_by_ligand(docked, tmp_path):
    _, molecules, _ = docked
    path = tmp_path / "docked.sdf"
    write_poses(molecules, path)
    records = [r for r in Chem.SDMolSupplier(str(path))]
    assert len(records) == sum(m.n_confs for m in molecules)
    assert float(records[0].GetProp("vina_score")) < 0 and records[0].GetProp("pose_rank") == "1"
    again = read_sdf(path)
    assert [m.name for m in again] == ["benzamidine", "benzylamine"]
    assert [m.n_confs for m in again] == [m.n_confs for m in molecules]
    result = analyse(again, align="none")
    assert result.similarity.shape == (2, 2)


def test_keep_poses_by_score(docked):
    mol = docked[1][0].mol
    kept = keep_poses(mol, 0.0)
    assert pose_scores(kept) == [min(pose_scores(mol))] * kept.GetNumConformers()


def test_box_contains():
    box = Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    assert box.contains([[4.9, -4.9, 0.0]]) and not box.contains([[5.1, 0.0, 0.0]])


def test_missing_binary_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        find_vina_binary(tmp_path / "no-vina")
