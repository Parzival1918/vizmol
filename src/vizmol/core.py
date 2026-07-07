"""Core visualization engine for molecular structures using the OVITO Python API.

Provides the :class:`MoleculeVisualizer` class for loading atomic structures,
generating bonds based on covalent radii, and rendering publication-quality
images and animations.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np

# Suppress the PyPI-in-Anaconda warning that OVITO emits
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

from ovito.io import import_file  # noqa: E402
from ovito.modifiers import CreateBondsModifier, ReplicateModifier  # noqa: E402
from ovito.data import BondsEnumerator  # noqa: E402
from ovito.vis import TachyonRenderer, Viewport  # noqa: E402
import ovito  # noqa: E402

# ---------------------------------------------------------------------------
# Covalent radii (Å) – Cordero et al., Dalton Trans. (2008) 2832–2838
# Indexed by upper-cased element symbol.
# ---------------------------------------------------------------------------
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "HE": 0.28,
    "LI": 1.28, "BE": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "NE": 0.58,
    "NA": 1.66, "MG": 1.41, "AL": 1.21, "SI": 1.11, "P": 1.07,
    "S": 1.05, "CL": 1.02, "AR": 1.06,
    "K": 2.03, "CA": 1.76, "SC": 1.70, "TI": 1.60, "V": 1.53,
    "CR": 1.39, "MN": 1.39, "FE": 1.32, "CO": 1.26, "NI": 1.24,
    "CU": 1.32, "ZN": 1.22, "GA": 1.22, "GE": 1.20, "AS": 1.19,
    "SE": 1.20, "BR": 1.20, "KR": 1.16,
    "RB": 2.20, "SR": 1.95, "Y": 1.90, "ZR": 1.75, "NB": 1.64,
    "MO": 1.54, "TC": 1.47, "RU": 1.46, "RH": 1.42, "PD": 1.39,
    "AG": 1.45, "CD": 1.44, "IN": 1.42, "SN": 1.39, "SB": 1.39,
    "TE": 1.38, "I": 1.39, "XE": 1.40,
    "CS": 2.44, "BA": 2.15, "LA": 2.07, "CE": 2.04, "PR": 2.03,
    "ND": 2.01, "PM": 1.99, "SM": 1.98, "EU": 1.98, "GD": 1.96,
    "TB": 1.94, "DY": 1.92, "HO": 1.92, "ER": 1.89, "TM": 1.90,
    "YB": 1.87, "LU": 1.87,
    "HF": 1.75, "TA": 1.70, "W": 1.62, "RE": 1.51, "OS": 1.44,
    "IR": 1.41, "PT": 1.36, "AU": 1.36, "HG": 1.32, "TL": 1.45,
    "PB": 1.46, "BI": 1.48, "PO": 1.40, "AT": 1.50, "RN": 1.50,
    "FR": 2.60, "RA": 2.21, "AC": 2.15, "TH": 2.06, "PA": 2.00,
    "U": 1.96, "NP": 1.90, "PU": 1.87, "AM": 1.80, "CM": 1.69,
}

# Default fallback covalent radius for elements not in the table (Å)
_DEFAULT_RADIUS: float = 1.50

# ---------------------------------------------------------------------------
# Rendering style presets
# ---------------------------------------------------------------------------
_STYLE_PRESETS: dict[str, dict] = {
    "realistic": {
        "ambient_occlusion": True,
        "ambient_occlusion_brightness": 0.8,
        "shadows": True,
        "direct_light_intensity": 0.9,
        "antialiasing": True,
        "antialiasing_samples": 12,
    },
    "cartoon": {
        "ambient_occlusion": False,
        "shadows": False,
        "direct_light_intensity": 1.0,
        "antialiasing": True,
        "antialiasing_samples": 12,
    },
}

# ---------------------------------------------------------------------------
# Representation presets (particle_scale, bond_radius)
# ---------------------------------------------------------------------------
_REPRESENTATION_PRESETS: dict[str, dict] = {
    "ball-and-stick": {"particle_scale": 0.4, "bond_radius": 0.15},
    "space-filling": {"particle_scale": 1.0, "bond_radius": 0.0},
    "wireframe": {"particle_scale": 0.15, "bond_radius": 0.08},
}


# ---------------------------------------------------------------------------
# Molecule-wholeness modifier (BFS on bond graph using PBC shift vectors)
# ---------------------------------------------------------------------------

def _make_molecules_whole(frame: int, data) -> None:  # noqa: ANN001, ARG001
    """Custom OVITO modifier that shifts atoms so that every molecule is
    geometrically contiguous (not split across periodic boundaries).

    The algorithm performs a BFS traversal of the bond graph.  For each bond
    that crosses a periodic boundary, the ``Periodic Image`` shift vector tells
    us by how many cell vectors the bonded neighbour is displaced.  We
    accumulate these shifts and apply them to the Cartesian positions.
    """
    bonds = data.particles.bonds
    if bonds is None or bonds.count == 0:
        return

    topology = bonds["Topology"]
    pbc_shift = bonds["Periodic Image"]
    cell_matrix = np.array(data.cell[:3, :3])  # 3×3 cell vectors

    n = data.particles.count
    visited = np.zeros(n, dtype=bool)
    shifts = np.zeros((n, 3), dtype=float)

    enum = BondsEnumerator(bonds)

    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True

        while queue:
            current = queue.popleft()
            for bond_idx in enum.bonds_of_particle(current):
                a, b = topology[bond_idx]
                pbc = pbc_shift[bond_idx]
                if a == current:
                    neighbor, shift = b, pbc
                else:
                    neighbor, shift = a, -pbc
                if not visited[neighbor]:
                    visited[neighbor] = True
                    shifts[neighbor] = shifts[current] + shift
                    queue.append(neighbor)

    # Apply accumulated lattice-vector shifts to Cartesian positions
    particles = data.particles_
    positions = particles.positions_
    positions[:] += shifts @ cell_matrix

    # Zero out the PBC shift vectors on bonds (molecules are now whole)
    bonds_mut = data.particles_.bonds_
    pbc_prop = bonds_mut["Periodic Image_"]
    pbc_prop[:] = 0


class MoleculeVisualizer:
    """High-level wrapper around the OVITO pipeline for molecular visualisation.

    Parameters
    ----------
    file_path : str | Path
        Path to an atomic-structure file readable by OVITO (XYZ, CIF, POSCAR,
        LAMMPS dump, PDB, …).
    bond_padding : float
        Extra distance (Å) added to the sum of covalent radii when deciding
        whether two atoms are bonded.  Default is ``0.3``.
    style : ``"realistic"`` | ``"cartoon"``
        Rendering style preset.  Default is ``"realistic"``.
    representation : ``"ball-and-stick"`` | ``"space-filling"`` | ``"wireframe"``
        Atom / bond representation preset.  Default is ``"ball-and-stick"``.
    supercell : tuple[int, int, int] | None
        If given, replicate the unit cell along (a, b, c) to show periodic
        images.  For example ``(2, 2, 2)`` produces a 2×2×2 supercell.
    show_cell : bool
        Whether to render the simulation-cell wireframe.  Default is ``True``
        for periodic systems and ``False`` otherwise.
    """

    def __init__(
        self,
        file_path: str | Path,
        bond_padding: float = 0.3,
        style: Literal["realistic", "cartoon"] = "realistic",
        representation: Literal[
            "ball-and-stick", "space-filling", "wireframe"
        ] = "ball-and-stick",
        supercell: tuple[int, int, int] | None = None,
        show_cell: bool | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        self.bond_padding = bond_padding
        self.style = style
        self.representation = representation
        self.supercell = supercell

        # Load the pipeline
        self._pipeline = import_file(str(self.file_path))

        # Detect whether the structure is periodic
        data = self._pipeline.compute()
        self._is_periodic = (
            data.cell is not None and any(data.cell.pbc)
        )

        if show_cell is None:
            self.show_cell = self._is_periodic
        else:
            self.show_cell = show_cell

        # Apply bond generation
        self._apply_bonds()

        # For periodic systems: make molecules whole before any replication
        if self._is_periodic:
            self._pipeline.modifiers.append(_make_molecules_whole)

        # Supercell replication (after making molecules whole)
        if self.supercell is not None:
            nx, ny, nz = self.supercell
            self._pipeline.modifiers.append(
                ReplicateModifier(num_x=nx, num_y=ny, num_z=nz)
            )

        # Apply visual styling (including cell visibility)
        self._apply_style()

    # ------------------------------------------------------------------
    # Bond generation
    # ------------------------------------------------------------------

    def _apply_bonds(self) -> None:
        """Add a :class:`CreateBondsModifier` in *Pairwise* mode with cutoffs
        derived from covalent radii plus the configured padding."""
        # Compute once to discover particle types
        data = self._pipeline.compute()
        particle_types = data.particles.particle_types

        modifier = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)

        for pt_a in particle_types.types:
            r_a = COVALENT_RADII.get(pt_a.name.upper(), _DEFAULT_RADIUS)
            for pt_b in particle_types.types:
                r_b = COVALENT_RADII.get(pt_b.name.upper(), _DEFAULT_RADIUS)
                cutoff = r_a + r_b + self.bond_padding
                modifier.set_pairwise_cutoff(pt_a.name, pt_b.name, cutoff)

        self._bond_modifier = modifier
        self._pipeline.modifiers.append(modifier)

    # ------------------------------------------------------------------
    # Visual styling
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        """Configure visual elements (particle sizes, bond radii, shading)
        according to the chosen *representation* and *style* presets."""
        rep = _REPRESENTATION_PRESETS[self.representation]

        # Particles
        data = self._pipeline.compute()
        particles_vis = data.particles.vis
        particles_vis.scaling = rep["particle_scale"]

        # Bonds
        bond_radius = rep["bond_radius"]
        self._bond_modifier.vis.width = bond_radius * 2
        self._bond_modifier.vis.enabled = bond_radius > 0

        # Cartoon-specific: flat shading on bonds
        if self.style == "cartoon":
            self._bond_modifier.vis.flat_shading = True

        # Simulation cell wireframe visibility
        if data.cell is not None and data.cell.vis is not None:
            data.cell.vis.enabled = self.show_cell

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------

    def _make_renderer(self) -> TachyonRenderer:
        """Return a :class:`TachyonRenderer` configured for the active style."""
        preset = _STYLE_PRESETS[self.style]
        return TachyonRenderer(**preset)

    def _make_viewport(self) -> Viewport:
        """Return a :class:`Viewport` zoomed to fit the current scene."""
        self._pipeline.add_to_scene()
        vp = Viewport(type=Viewport.Type.Perspective)
        vp.zoom_all()
        return vp

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------

    @property
    def num_bonds(self) -> int:
        """Number of bonds in the current pipeline."""
        data = self._pipeline.compute()
        if data.particles.bonds is not None:
            return data.particles.bonds.count
        return 0

    @property
    def num_particles(self) -> int:
        """Number of particles (atoms) in the current pipeline."""
        return self._pipeline.compute().particles.count

    def render_image(
        self,
        output: str | Path,
        *,
        width: int = 800,
        height: int = 600,
    ) -> Path:
        """Render a static image of the structure.

        Parameters
        ----------
        output : str | Path
            Destination file path (PNG or JPEG based on extension).
        width, height : int
            Image dimensions in pixels.

        Returns
        -------
        Path
            The resolved output path.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        vp = self._make_viewport()
        renderer = self._make_renderer()
        vp.render_image(
            filename=str(output),
            size=(width, height),
            renderer=renderer,
        )
        self._pipeline.remove_from_scene()
        return output

    def render_animation(
        self,
        output: str | Path,
        *,
        width: int = 800,
        height: int = 600,
        num_frames: int = 60,
        fps: int = 30,
        animation_type: Literal["rotate"] = "rotate",
    ) -> Path:
        """Render an animation (e.g. a 360° rotation) of the structure.

        Parameters
        ----------
        output : str | Path
            Destination file path (AVI or MP4 based on extension).
        width, height : int
            Frame dimensions in pixels.
        num_frames : int
            Total number of frames to render.
        fps : int
            Frames per second.
        animation_type : str
            Currently only ``"rotate"`` is supported (360° Y-axis rotation).

        Returns
        -------
        Path
            The resolved output path.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if animation_type != "rotate":
            raise ValueError(
                f"Unsupported animation type: {animation_type!r}. "
                "Only 'rotate' is currently supported."
            )

        # Add a custom modifier that rotates all particles *and* the
        # simulation cell around the Y axis.
        def _rotate(frame: int, data) -> None:  # noqa: ANN001
            angle = 2 * math.pi * frame / num_frames
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rot = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a],
            ])

            # Rotate particle positions around their centre of mass
            particles = data.particles_
            positions = particles.positions_
            center = np.mean(positions, axis=0)
            positions[:] = (positions - center) @ rot.T + center

            # Rotate the simulation cell (if present)
            if data.cell is not None:
                cell = data.cell_
                matrix = np.array(cell.matrix)
                # Rotate the three cell column-vectors (first 3 cols)
                cell_vecs = matrix[:3, :3]
                origin = matrix[:3, 3]
                new_vecs = rot @ cell_vecs
                new_origin = rot @ (origin - center) + center
                new_matrix = np.zeros((3, 4))
                new_matrix[:3, :3] = new_vecs
                new_matrix[:3, 3] = new_origin
                cell.matrix = new_matrix

        self._pipeline.modifiers.append(_rotate)

        vp = self._make_viewport()
        renderer = self._make_renderer()

        # Set the animation interval AFTER add_to_scene (called inside
        # _make_viewport) and disable auto-adjust so OVITO does not reset
        # the frame range to match the pipeline's single source frame.
        ovito.scene.anim.auto_adjust_interval = False
        ovito.scene.anim.first_frame = 0
        ovito.scene.anim.last_frame = num_frames - 1
        ovito.scene.anim.frames_per_second = fps

        vp.render_anim(
            filename=str(output),
            size=(width, height),
            renderer=renderer,
        )

        # Clean up: remove the rotation modifier (last added) and take the
        # pipeline off scene.  Re-enable auto-adjust for future operations.
        ovito.scene.anim.auto_adjust_interval = True
        del self._pipeline.modifiers[-1]
        self._pipeline.remove_from_scene()
        return output
