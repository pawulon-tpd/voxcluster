"""voxcluster demo: cluster molecules by how they occupy space.

Paste SMILES (3D conformers are generated with RDKit ETKDG) or upload an SDF of
poses. The app aligns everything onto a reference molecule, voxelises each
molecule's conformer ensemble into pharmacophore channels, clusters molecules
by occupancy overlap, and draws the voxels in 3D. With a receptor, the SMILES
are docked with AutoDock Vina instead and the poses are compared in place.
"""

import html
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from voxorcist.voxelize import CHANNELS

from voxcluster.analysis import CHANNEL_SETS, Analysis, analyse
from voxcluster.conformers import from_smiles, parse_smiles, read_sdf
from voxcluster.docking import backend, box_from_ligand, dock, pose_scores

EXAMPLES = Path(__file__).parent / "examples"
EXAMPLE = (EXAMPLES / "molecules.smi").read_text()
TRYPSIN = (EXAMPLES / "trypsin_ligands.smi").read_text()
POCKET, POCKET_LIGAND = EXAMPLES / "3ptb" / "pocket.pdb", EXAMPLES / "3ptb" / "ligand.sdf"
VIEWER_JS = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
AUTO = "auto (largest molecule)"
MAX_MOLECULES = 60
MAX_DOCKED = 20
PROGRESS = gr.Progress()  # Gradio injects a tracker for parameters defaulting to one
MAX_CONFS_SHOWN = 8
COLOR_A, COLOR_B, COLOR_BOTH = "#1c7ed6", "#e8590c", "#7048e8"
TABLE_COLUMNS = ["name", "cluster", "vina_best", "n_confs", "envelope_A3", "core_A3", "core_fraction",
                 "shape", "shape_tanimoto_to_ref", "smiles"]


def load_molecules(smiles_text: str, sdf, n_confs: int, docking: tuple, cache: dict,
                   progress=None):
    """Molecules from the SDF if given, else docked into the receptor if one
    is given, else conformers from the SMILES text. Cached per input so
    changing only analysis options is fast."""
    key = (sdf, smiles_text, n_confs, docking)
    if cache.get("key") == key:
        return cache["molecules"], cache["skipped"]
    skipped = []
    receptor, box_ligand, size, exhaustiveness, n_poses = docking
    if sdf:
        molecules = read_sdf(sdf)
    elif receptor:
        if not box_ligand:
            raise gr.Error("Docking needs a box ligand (SDF/MOL2) to centre the search box.")
        try:
            backend()
        except FileNotFoundError as err:
            raise gr.Error(str(err))
        ligands = parse_smiles(smiles_text)[:MAX_DOCKED]
        done = []

        def log(msg):
            done.append(msg)
            if progress is not None:
                progress(len(done) / len(ligands), desc=msg)

        molecules, skipped = dock(receptor, ligands, box_from_ligand(box_ligand, float(size)),
                                  int(exhaustiveness), int(n_poses), log=log)
    else:
        molecules = []
        for smiles, name in parse_smiles(smiles_text)[:MAX_MOLECULES]:
            try:
                molecules.append(from_smiles(smiles, name, int(n_confs)))
            except ValueError as err:
                skipped.append(str(err))
    cache.update(key=key, molecules=molecules, skipped=skipped)
    return molecules, skipped


def heatmap(result: Analysis):
    order = result.order
    names = [result.names[i] for i in order]
    sim = result.similarity[np.ix_(order, order)]
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.6 + 0.42 * n, 1.2 + 0.42 * n), dpi=110)
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    labels = result.labels[order]
    for k in range(1, n):  # cluster boundaries in dendrogram order
        if labels[k] != labels[k - 1]:
            ax.axhline(k - 0.5, color="white", lw=1.5)
            ax.axvline(k - 0.5, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="occupancy similarity")
    fig.tight_layout()
    return fig


def run(smiles_text, sdf, n_confs, reference, align, features, min_similarity,
        receptor, box_ligand, box_size, exhaustiveness, n_poses, cache,
        progress=PROGRESS):
    docking = (receptor, box_ligand, box_size, exhaustiveness, n_poses)
    molecules, skipped = load_molecules(smiles_text, sdf, n_confs, docking, cache, progress)
    if not molecules:
        empty = gr.update(choices=[AUTO], value=AUTO)
        msg = "No valid molecules." + ("\n\nSkipped: " + "; ".join(skipped) if skipped else "")
        return (msg, None, None, empty, gr.update(), gr.update(), cache)
    names = [m.name for m in molecules]
    ref = names.index(reference) if reference in names else None
    docked = all(m.source == "docking" for m in molecules)
    keep = docked or align == "keep coordinates"
    result = analyse(molecules, ref, "none" if keep else "shape", features, min_similarity)
    cache["result"] = result
    cache["receptor"] = Path(receptor).read_text() if docked else None
    n_clusters = len(set(result.labels))
    summary = (f"**{len(molecules)} molecules → {n_clusters} clusters** "
               f"(average linkage, similarity ≥ {min_similarity:.2f} on `{features}`). "
               f"Reference: **{names[result.reference]}**; shared box {result.frame.spec.size:g} Å.")
    if docked:
        summary += (f"\n\nDocked with AutoDock Vina ({int(n_poses)} poses, exhaustiveness "
                    f"{int(exhaustiveness)}); poses kept in the receptor frame. `vina_best` is "
                    "the best Vina score in kcal/mol (a ranking score, not an affinity).")
    if skipped:
        summary += "\n\nSkipped: " + "; ".join(skipped)
    result.table["vina_best"] = [min(pose_scores(m.mol)) if m.source == "docking"
                                 else float("nan") for m in molecules]
    columns = TABLE_COLUMNS if docked else [c for c in TABLE_COLUMNS if c != "vina_best"]
    table = result.table[columns].sort_values(["cluster", "name"]).round(2)
    pick = names[result.order[0]]
    return (summary, table, heatmap(result),
            gr.update(choices=[AUTO] + names, value=reference if reference in names else AUTO),
            gr.update(choices=names, value=pick),
            gr.update(choices=["(none)"] + names, value="(none)"),
            cache)


def voxel_js(occ: np.ndarray, center: np.ndarray, resolution: float, threshold: float,
             color: str | None = None) -> tuple[list[str], np.ndarray]:
    """addBox calls for voxels with occupancy >= threshold. Without a fixed
    colour, voxels are shaded by occupancy (light = rarely, dark = always)."""
    d = occ.shape[0]
    idx = np.argwhere(occ >= threshold)
    pos = center + (idx - (d - 1) / 2) * resolution
    calls = []
    for (i, j, k), p in zip(idx, pos):
        f = float(occ[i, j, k])
        c = color or f"rgb({int(190 - 165 * f)},{int(215 - 135 * f)},{int(245 - 65 * f)})"
        calls.append(f"v.addBox({{center:{{x:{p[0]:.2f},y:{p[1]:.2f},z:{p[2]:.2f}}},"
                     f"dimensions:{{w:{resolution * 0.92},h:{resolution * 0.92},d:{resolution * 0.92}}},"
                     f"color:'{c}',opacity:{0.15 + 0.55 * f:.2f}}});")
    return calls, idx


def conformer_js(mol: Chem.Mol, color: str) -> list[str]:
    calls = []
    for conf in list(mol.GetConformers())[:MAX_CONFS_SHOWN]:
        block = Chem.MolToMolBlock(Chem.RemoveHs(mol), confId=conf.GetId())
        calls.append(f"v.addModel({json.dumps(block)},'sdf').setStyle({{}},"
                     f"{{stick:{{radius:0.14,color:'{color}',opacity:0.8}}}});")
    return calls


def pocket_js(pdb: str, frame) -> list[str]:
    """Receptor residues with an atom inside the shared box, as thin grey
    sticks (polar H hidden)."""
    half = frame.spec.size / 2
    residues: dict[str, list[str]] = {}
    keep = set()
    for line in pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res = line[17:27]
        residues.setdefault(res, []).append(line[:66])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if (np.abs(xyz - frame.center) <= half).all():
            keep.add(res)
    text = "\n".join(line for res in residues if res in keep for line in residues[res])
    return [(f"v.addModel({json.dumps(text)},'pdb').setStyle({{}},"
             "{stick:{radius:0.07,color:'#adb5bd'}});v.setStyle({elem:'H'},{});")]


def page(js: list[str]) -> str:
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><script src="{VIEWER_JS}"></script>'
           "<style>html,body{margin:0;height:100%;background:#fff}#v{width:100%;height:100%}</style>"
           '</head><body><div id="v"></div><script>'
           'const v=$3Dmol.createViewer("v",{backgroundColor:"white"});'
           + "".join(js) + "v.zoomTo();v.zoom(0.55);v.render();</script></body></html>")
    return (f'<iframe srcdoc="{html.escape(doc, quote=True)}" '
            'style="width:100%;height:560px;border:1px solid #dee2e6;border-radius:8px"></iframe>')


def view(name_a, name_b, mode, channel, threshold, cache):
    result: Analysis | None = cache.get("result")
    if result is None or name_a not in result.names:
        return "<p>Run an analysis first.</p>", ""
    c = CHANNELS.index(channel)
    a = result.names.index(name_a)
    v = result.frame.voxel_volume
    if mode == "envelope (own frame)":
        mol, occ, frame = result.own[a]
        calls, idx = voxel_js(occ[c], frame.center, frame.spec.resolution, threshold)
        calls += conformer_js(mol, "#495057")
        note = (f"**{name_a}**, `{channel}`: {len(idx) * v:.0f} Å³ occupied by ≥ {threshold:.0%} "
                f"of {mol.GetNumConformers()} conformers. Darker = occupied more often.")
        return page(calls), note
    occ_a = result.occupancies[a][c] >= threshold
    calls = pocket_js(cache["receptor"], result.frame) if cache.get("receptor") else []
    calls += conformer_js(result.aligned[a], COLOR_A)
    if name_b in result.names:
        b = result.names.index(name_b)
        occ_b = result.occupancies[b][c] >= threshold
        only_a, only_b, both = occ_a & ~occ_b, occ_b & ~occ_a, occ_a & occ_b
        for mask, color in ((only_a, COLOR_A), (only_b, COLOR_B), (both, COLOR_BOTH)):
            calls += voxel_js(mask.astype(np.float32), result.frame.center,
                              result.frame.spec.resolution, 0.5, color)[0]
        calls += conformer_js(result.aligned[b], COLOR_B)
        note = (f"Shared frame, `{channel}`, voxels occupied by ≥ {threshold:.0%} of conformers: "
                f"<span style='color:{COLOR_A}'>■</span> only {name_a} {only_a.sum() * v:.0f} Å³, "
                f"<span style='color:{COLOR_B}'>■</span> only {name_b} {only_b.sum() * v:.0f} Å³, "
                f"<span style='color:{COLOR_BOTH}'>■</span> both {both.sum() * v:.0f} Å³. "
                f"Similarity {result.similarity[a, b]:.2f}.")
    else:
        calls += voxel_js(occ_a.astype(np.float32), result.frame.center,
                          result.frame.spec.resolution, 0.5, COLOR_A)[0]
        note = f"**{name_a}** in the shared frame: {occ_a.sum() * v:.0f} Å³ (`{channel}`)."
    return page(calls), note


with gr.Blocks(title="voxcluster") as demo:
    cache = gr.State({})
    gr.Markdown("# voxcluster\nCluster molecules by the space they occupy: RDKit conformers → "
                "shape alignment → pharmacophore voxels → similarity → clusters.")
    with gr.Tab("Analyse"), gr.Row():
        with gr.Column(scale=1):
            smiles = gr.Textbox(EXAMPLE, lines=12, label="SMILES (one per line: SMILES name)")
            sdf = gr.File(label="…or an SDF of 3D poses (overrides SMILES)", file_types=[".sdf"])
            n_confs = gr.Slider(5, 100, value=30, step=5, label="Conformers per SMILES")
            reference = gr.Dropdown([AUTO], value=AUTO, label="Reference molecule (sets the frame)")
            align = gr.Radio(["shape-align to reference", "keep coordinates"],
                             value="shape-align to reference", label="Alignment")
            features = gr.Radio(sorted(CHANNEL_SETS), value="shape+features",
                                label="Compare on")
            min_sim = gr.Slider(0.1, 0.95, value=0.5, step=0.05,
                                label="Cluster cut-off (min. similarity)")
            with gr.Accordion("Dock into a receptor (AutoDock Vina, optional)", open=False):
                gr.Markdown("With a receptor, the SMILES are docked with Vina and the poses "
                            "are compared in the receptor frame (alignment is ignored). "
                            f"**Slow:** seconds to a minute per ligand; at most {MAX_DOCKED} "
                            "ligands.")
                receptor = gr.File(label="Receptor (PDB or PDBQT)", file_types=[".pdb", ".pdbqt"])
                box_ligand = gr.File(label="Box ligand: search box centred on it",
                                     file_types=[".sdf", ".mol", ".mol2", ".pdb"])
                box_size = gr.Slider(10, 40, value=20, step=2, label="Box edge (Å)")
                exhaustiveness = gr.Slider(1, 32, value=8, step=1, label="Exhaustiveness")
                n_poses = gr.Slider(1, 20, value=9, step=1, label="Poses per ligand")
                trypsin = gr.Button("Load trypsin example (PDB 3PTB)", size="sm")
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown()
            plot = gr.Plot(label="Similarity (dendrogram order, white lines = clusters)")
            table = gr.Dataframe(label="Molecules", wrap=True)
    with gr.Tab("3D voxels"), gr.Row():
        with gr.Column(scale=1):
            mol_a = gr.Dropdown([], label="Molecule A")
            mol_b = gr.Dropdown(["(none)"], value="(none)", label="Molecule B (shared frame only)")
            mode = gr.Radio(["envelope (own frame)", "shared frame"],
                            value="envelope (own frame)", label="View")
            channel = gr.Dropdown(list(CHANNELS), value="excluded_volume", label="Channel")
            threshold = gr.Slider(0.05, 1.0, value=0.1, step=0.05,
                                  label="Show voxels occupied by ≥ this fraction of conformers")
        with gr.Column(scale=2):
            view3d = gr.HTML()
            note = gr.Markdown()

    go.click(run, [smiles, sdf, n_confs, reference, align, features, min_sim,
                   receptor, box_ligand, box_size, exhaustiveness, n_poses, cache],
             [summary, table, plot, reference, mol_a, mol_b, cache])
    trypsin.click(lambda: (TRYPSIN, str(POCKET), str(POCKET_LIGAND), None),
                  outputs=[smiles, receptor, box_ligand, sdf])
    view_controls = [mol_a, mol_b, mode, channel, threshold]
    for control in view_controls:
        control.change(view, view_controls + [cache], [view3d, note])

if __name__ == "__main__":
    demo.launch()
