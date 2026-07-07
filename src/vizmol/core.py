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

# Van der Waals radii (Å) - standard Alvarez (2013) values
_VDW_RADII: dict[str, float] = {
    "H": 1.20, "HE": 1.40, "LI": 1.82, "BE": 1.53, "B": 1.92,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "NE": 1.54,
    "NA": 2.27, "MG": 1.73, "AL": 1.84, "SI": 2.10, "P": 1.80,
    "S": 1.80, "CL": 1.75, "AR": 1.88, "K": 2.75, "CA": 2.31,
    "CU": 1.40, "ZN": 1.39, "BR": 1.85, "AG": 1.72, "I": 1.98,
    "PT": 1.72, "AU": 1.66
}
_DEFAULT_VDW_RADIUS: float = 2.0

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
    "flat": {
        "ambient_occlusion": False,
        "shadows": False,
        "direct_light_intensity": 1.0,
        "antialiasing": True,
        "antialiasing_samples": 4,
    },
    "hand-drawn": {},
}

# ---------------------------------------------------------------------------
# Representation presets (particle_scale, bond_radius)
# ---------------------------------------------------------------------------
_REPRESENTATION_PRESETS: dict[str, dict] = {
    "ball-and-stick": {"particle_scale": 0.4, "bond_radius": 0.15},
    "space-filling": {"particle_scale": 1.0, "bond_radius": 0.0},
    "wireframe": {"particle_scale": 0, "bond_radius": 0.2},
    "wireframe-thin": {"particle_scale": 0, "bond_radius": 0.1},
    "uniform": {"particle_scale": 1.0, "bond_radius": 0.15},
    "tube": {"particle_scale": 0.01, "bond_radius": 0.25},
    "wire": {"particle_scale": 0.01, "bond_radius": 0.10},
    "vdw": {"particle_scale": 1.0, "bond_radius": 0.0},
    "pmol": {"particle_scale": 0.4, "bond_radius": 0.15},
    "paton": {"particle_scale": 0.4, "bond_radius": 0.15},
    "skeletal": {"particle_scale": 0.25, "bond_radius": 0.14},
    "vdw-overlay": {"particle_scale": 0.25, "bond_radius": 0.1},
}

# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------
_COLOR_SCHEMES: dict[str, dict[str, tuple[float, float, float]]] = {
    "default": {},
    "muted": {
        "C": (0.6, 0.6, 0.6),
        "H": (0.9, 0.9, 0.9),
        "O": (0.8, 0.4, 0.4),
        "N": (0.4, 0.6, 0.8),
        "S": (0.8, 0.8, 0.4),
        "P": (0.8, 0.6, 0.4),
        "CL": (0.6, 0.8, 0.6),
        "F": (0.6, 0.8, 0.8),
        "BR": (0.7, 0.5, 0.4),
        "I": (0.6, 0.4, 0.6),
    },
    "paton": {
        "C": (0.851, 0.851, 0.851),
        "H": (0.980, 0.980, 0.980),
        "N": (0.498, 0.498, 0.749),
    },
    "pmol": {
        "C": (0.851, 0.851, 0.851),
        "H": (0.980, 0.980, 0.980),
        "N": (0.498, 0.498, 0.749),
    },
    "xyzrender": {
        'H': (1.0, 1.0, 1.0), 'HE': (0.851, 1.0, 1.0), 'LI': (0.8, 0.502, 1.0), 'BE': (0.761, 1.0, 0.0), 'B': (1.0, 0.71, 0.71), 'C': (0.565, 0.565, 0.565), 'N': (0.188, 0.314, 0.973), 'O': (1.0, 0.051, 0.051), 'F': (0.565, 0.878, 0.314), 'NE': (0.702, 0.89, 0.961), 'NA': (0.671, 0.361, 0.949), 'MG': (0.541, 1.0, 0.0), 'AL': (0.749, 0.651, 0.651), 'SI': (0.941, 0.784, 0.627), 'P': (1.0, 0.502, 0.0), 'S': (1.0, 1.0, 0.188), 'CL': (0.122, 0.941, 0.122), 'AR': (0.502, 0.82, 0.89), 'K': (0.561, 0.251, 0.831), 'CA': (0.239, 1.0, 0.0), 'SC': (0.902, 0.902, 0.902), 'TI': (0.749, 0.761, 0.78), 'V': (0.651, 0.651, 0.671), 'CR': (0.541, 0.6, 0.78), 'MN': (0.612, 0.478, 0.78), 'FE': (0.878, 0.4, 0.2), 'CO': (0.941, 0.565, 0.627), 'NI': (0.314, 0.816, 0.314), 'CU': (0.784, 0.502, 0.2), 'ZN': (0.49, 0.502, 0.69), 'GA': (0.761, 0.561, 0.561), 'GE': (0.4, 0.561, 0.561), 'AS': (0.741, 0.502, 0.89), 'SE': (1.0, 0.631, 0.0), 'BR': (0.651, 0.161, 0.161), 'KR': (0.361, 0.722, 0.82), 'RB': (0.439, 0.18, 0.69), 'SR': (0.0, 1.0, 0.0), 'Y': (0.58, 1.0, 1.0), 'ZR': (0.58, 0.878, 0.878), 'NB': (0.451, 0.761, 0.788), 'MO': (0.329, 0.71, 0.71), 'TC': (0.231, 0.62, 0.62), 'RU': (0.141, 0.561, 0.561), 'RH': (0.039, 0.49, 0.549), 'PD': (0.0, 0.412, 0.522), 'AG': (0.753, 0.753, 0.753), 'CD': (1.0, 0.851, 0.561), 'IN': (0.651, 0.459, 0.451), 'SN': (0.4, 0.502, 0.502), 'SB': (0.62, 0.388, 0.71), 'TE': (0.831, 0.478, 0.0), 'I': (0.58, 0.0, 0.58), 'XE': (0.259, 0.62, 0.69), 'CS': (0.341, 0.09, 0.561), 'BA': (0.0, 0.788, 0.0), 'LA': (0.439, 0.831, 1.0), 'CE': (1.0, 1.0, 0.78), 'PR': (0.851, 1.0, 0.78), 'ND': (0.78, 1.0, 0.78), 'PM': (0.639, 1.0, 0.78), 'SM': (0.561, 1.0, 0.78), 'EU': (0.38, 1.0, 0.78), 'GD': (0.271, 1.0, 0.78), 'TB': (0.188, 1.0, 0.78), 'DY': (0.122, 1.0, 0.78), 'HO': (0.0, 1.0, 0.612), 'ER': (0.0, 0.902, 0.459), 'TM': (0.0, 0.831, 0.322), 'YB': (0.0, 0.749, 0.22), 'LU': (0.0, 0.671, 0.141), 'HF': (0.302, 0.761, 1.0), 'TA': (0.302, 0.651, 1.0), 'W': (0.129, 0.58, 0.839), 'RE': (0.149, 0.49, 0.671), 'OS': (0.149, 0.4, 0.588), 'IR': (0.09, 0.329, 0.529), 'PT': (0.816, 0.816, 0.878), 'AU': (1.0, 0.82, 0.137), 'HG': (0.722, 0.722, 0.816), 'TL': (0.651, 0.329, 0.302), 'PB': (0.341, 0.349, 0.38), 'BI': (0.62, 0.31, 0.71), 'PO': (0.671, 0.361, 0.0), 'AT': (0.459, 0.31, 0.271), 'RN': (0.259, 0.51, 0.588), 'FR': (0.259, 0.0, 0.4), 'RA': (0.0, 0.49, 0.0), 'AC': (0.439, 0.671, 0.98), 'TH': (0.0, 0.729, 1.0), 'PA': (0.0, 0.631, 1.0), 'U': (0.0, 0.561, 1.0), 'NP': (0.0, 0.502, 1.0), 'PU': (0.0, 0.42, 1.0), 'AM': (0.329, 0.361, 0.949), 'CM': (0.471, 0.361, 0.89), 'BK': (0.541, 0.31, 0.89), 'CF': (0.631, 0.212, 0.831), 'ES': (0.702, 0.122, 0.831), 'FM': (0.702, 0.122, 0.729), 'MD': (0.702, 0.051, 0.651), 'NO': (0.741, 0.051, 0.529), 'LR': (0.78, 0.0, 0.4), 'RF': (0.8, 0.0, 0.349), 'DB': (0.627, 0.627, 0.627)
    }
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


# ---------------------------------------------------------------------------
# Rotation matrix around an arbitrary axis (Rodrigues' formula)
# ---------------------------------------------------------------------------

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the 3×3 rotation matrix for *angle* radians about *axis*.

    Uses Rodrigues' rotation formula.  *axis* must be a unit vector.
    """
    k = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


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
    style : ``"realistic"`` | ``"cartoon"`` | ``"hand-drawn"``
        Rendering style preset.  Default is ``"realistic"``.
    representation : ``"ball-and-stick"`` | ``"space-filling"`` | ``"wireframe"``
        Atom / bond representation preset.  Default is ``"ball-and-stick"``.
    supercell : tuple[int, int, int] | None
        If given, replicate the unit cell along (a, b, c) to show periodic
        images.  For example ``(2, 2, 2)`` produces a 2×2×2 supercell.
    show_cell : bool
        Whether to render the simulation-cell wireframe.  Default is ``True``
        for periodic systems and ``False`` otherwise.
    camera_azimuth : float
        Horizontal camera angle in degrees (0 = +X, 90 = +Z).  Default ``45``.
    camera_elevation : float
        Vertical camera angle in degrees (0 = horizontal, 90 = top-down).
        Default ``30``.
    camera_distance : float | None
        Distance from the focal point to the camera in Å.  ``None`` (default)
        auto-computes a distance that fits the whole structure. For
        orthographic projection, this sets the field-of-view width.
    focal_point : tuple[float, float, float] | None
        The 3-D point the camera looks at.  ``None`` (default) uses the
        centre of mass of the atoms.
    projection : ``"perspective"`` | ``"orthographic"``
        Camera projection type. Default is ``"orthographic"``.
    color_scheme : str
        Preset color scheme to use. Default is ``"default"``. Other built-in
        options include ``"muted"``.
    colors : dict[str, tuple[float, float, float]] | None
        Custom mapping of element symbols (e.g. ``"C"``, ``"O"``) to RGB
        tuples. Overrides ``color_scheme`` if provided.
    show_cell_axes : bool
        Whether to show a coordinate tripod representing the unit cell vectors
        (a, b, c). Default is ``False``.
    show_info : bool
        Whether to show an overlay text with molecular/crystal information.
        Default is ``False``.
    hide_hc_hydrogens : bool
        Whether to remove hydrogen atoms that are bonded to carbon atoms.
        Default is ``False``.
    extract_molecule : int | None
        Extract and render only a single molecule from the scene (by cluster ID).
        Default is ``None``.
    atom_borders : bool
        Whether to add black outlines to atoms. Default is ``False``.
    renderer : str | None
        Explicit rendering engine to use ('tachyon', 'ospray', 'opengl').
        Default is ``None`` (automatically chosen).
    quality : str
        Rendering quality level ('draft', 'medium', 'high').
        Default is ``high``.
    """

    def __init__(
        self,
        file_path: str | Path,
        bond_padding: float = 0.3,
        style: Literal["realistic", "cartoon", "flat", "hand-drawn"] = "realistic",
        representation: str = "ball-and-stick",
        supercell: tuple[int, int, int] | None = None,
        show_cell: bool | None = None,
        camera_azimuth: float = 45.0,
        camera_elevation: float = 30.0,
        camera_distance: float | None = None,
        focal_point: tuple[float, float, float] | None = None,
        projection: Literal["perspective", "orthographic"] = "orthographic",
        color_scheme: str = "default",
        colors: dict[str, tuple[float, float, float]] | None = None,
        show_cell_axes: bool = False,
        show_info: bool = False,
        hide_hc_hydrogens: bool = False,
        extract_molecule: int | None = None,
        atom_borders: bool = False,
        renderer: str | None = None,
        quality: Literal["draft", "medium", "high"] = "high",
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        self.bond_padding = bond_padding
        self.style = style
        self.representation = representation
        self.supercell = supercell
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_distance = camera_distance
        self.focal_point = focal_point
        self.projection = projection
        self.color_scheme = color_scheme
        self.colors = colors
        self.show_cell_axes = show_cell_axes
        self.show_info = show_info
        self.hide_hc_hydrogens = hide_hc_hydrogens
        self.extract_molecule = extract_molecule
        self.atom_borders = atom_borders
        self.renderer = renderer
        self.quality = quality
        self._vdw_modifier = None

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

        if self.extract_molecule is not None:
            from ovito.modifiers import ClusterAnalysisModifier, ExpressionSelectionModifier, DeleteSelectedModifier
            
            # Identify molecules (bonded clusters)
            self._pipeline.modifiers.append(ClusterAnalysisModifier(
                neighbor_mode=ClusterAnalysisModifier.NeighborMode.Bonding,
                sort_by_size=True,
            ))
            
            # Select everything except the requested molecule and delete it
            self._pipeline.modifiers.append(ExpressionSelectionModifier(
                expression=f"Cluster != {self.extract_molecule}"
            ))
            self._pipeline.modifiers.append(DeleteSelectedModifier())
            
            # Remove the simulation cell so it doesn't skew the camera zoom or render
            def _remove_cell(frame: int, data) -> None:  # noqa: ANN001
                if data.cell is not None:
                    data.objects.remove(data.cell)
            self._pipeline.modifiers.append(_remove_cell)
            
            # Override settings to hide cell since we deleted it
            self.show_cell = False
            self._is_periodic = False

        # Supercell replication (after making molecules whole)
        if self.supercell is not None and self.extract_molecule is None:
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

        # Optional: remove H bonded to C
        if self.hide_hc_hydrogens:
            def _remove_hc(frame: int, data) -> None:  # noqa: ANN001
                from ovito.data import BondsEnumerator
                bonds = data.particles.bonds
                if bonds is None: return
                ptypes = data.particles.particle_types
                topology = bonds.topology
                enum = BondsEnumerator(bonds)
                h_type = None
                c_type = None
                for pt in ptypes.types:
                    if pt.name.upper() == 'H': h_type = pt.id
                    elif pt.name.upper() == 'C': c_type = pt.id
                if h_type is None or c_type is None:
                    return
                to_delete = []
                for i in range(data.particles.count):
                    if ptypes[i] == h_type:
                        bonded_to_c = False
                        for bond_idx in enum.bonds_of_particle(i):
                            a, b = topology[bond_idx]
                            neighbor = b if a == i else a
                            if ptypes[neighbor] == c_type:
                                bonded_to_c = True
                                break
                        if bonded_to_c:
                            to_delete.append(i)
                if to_delete:
                    import numpy as np
                    sel = np.zeros(data.particles.count, dtype=int)
                    sel[to_delete] = 1
                    data.particles_.create_property('Selection', data=sel)
            
            from ovito.modifiers import DeleteSelectedModifier
            self._pipeline.modifiers.append(_remove_hc)
            self._pipeline.modifiers.append(DeleteSelectedModifier())

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
        if self.style in ("cartoon", "flat"):
            self._bond_modifier.vis.flat_shading = True
            
        if self.representation == "wireframe":
            # Force particles to have exactly the same radius as bonds
            particles_vis.scaling = 1.0
            
            def _set_wireframe_radii(frame: int, data) -> None:  # noqa: ANN001
                if data.particles is None or data.particles.particle_types is None:
                    return
                ptypes = data.particles_.particle_types_
                for pt in ptypes.types:
                    mut_pt = ptypes.make_mutable(pt)
                    mut_pt.radius = bond_radius

            self._pipeline.modifiers.append(_set_wireframe_radii)

        if self.representation == "uniform":
            self._bond_modifier.vis.coloring_mode = type(self._bond_modifier.vis).ColoringMode.Uniform
            self._bond_modifier.vis.color = (0.0, 0.0, 0.0)
            
            def _set_uniform_radii(frame: int, data) -> None:  # noqa: ANN001
                if data.particles is None or data.particles.particle_types is None:
                    return
                ptypes = data.particles_.particle_types_
                for pt in ptypes.types:
                    mut_pt = ptypes.make_mutable(pt)
                    mut_pt.radius = 0.5

            self._pipeline.modifiers.append(_set_uniform_radii)

        if self.representation == "vdw":
            def _set_vdw_radii(frame: int, data) -> None:  # noqa: ANN001
                if data.particles is None or data.particles.particle_types is None:
                    return
                ptypes = data.particles_.particle_types_
                for pt in ptypes.types:
                    mut_pt = ptypes.make_mutable(pt)
                    rad = _VDW_RADII.get(pt.name.upper(), _DEFAULT_VDW_RADIUS) if pt.name else _DEFAULT_VDW_RADIUS
                    mut_pt.radius = rad

            self._pipeline.modifiers.append(_set_vdw_radii)

        if self.representation == "vdw-overlay":
            def _vdw_overlay_mod(frame: int, data) -> None:  # noqa: ANN001
                from ovito.data import Particles
                import numpy as np

                if data.particles is None or data.particles.count == 0:
                    return

                vdw = Particles()
                vdw.identifier = "vdw"
                vdw.create_property("Position", data=data.particles.positions)
                vdw.create_property("Particle Type", data=data.particles.particle_types)
                vdw.create_property("Transparency", data=np.full(data.particles.count, 0.95))
                
                ptypes = data.particles.particle_types
                type_radii = {}
                for pt in ptypes.types:
                    type_radii[pt.id] = _VDW_RADII.get(pt.name.upper(), _DEFAULT_VDW_RADIUS) if pt.name else _DEFAULT_VDW_RADIUS
                radii = np.array([type_radii[t] for t in ptypes], dtype=float)
                vdw.create_property("Radius", data=radii)
                
                if "Color" in data.particles:
                    vdw.create_property("Color", data=data.particles.color)
                else:
                    colors = np.array([ptypes.type_by_id(t).color for t in ptypes], dtype=float)
                    vdw.create_property("Color", data=colors)
                
                vdw.vis.scaling = 1.0
                data.objects.append(vdw)

            self._vdw_modifier = _vdw_overlay_mod

        # Custom colors
        colors_dict = self.colors
        if colors_dict is None:
            if self.color_scheme not in _COLOR_SCHEMES:
                raise ValueError(f"Unknown color scheme: {self.color_scheme}")
            colors_dict = _COLOR_SCHEMES[self.color_scheme]

        if colors_dict:
            # Normalise keys to upper case
            norm_colors = {k.upper(): v for k, v in colors_dict.items()}
            
            def _set_colors(frame: int, data) -> None:  # noqa: ANN001
                if data.particles is None or data.particles.particle_types is None:
                    return
                ptypes = data.particles_.particle_types_
                for pt in ptypes.types:
                    sym = pt.name.upper()
                    if sym in norm_colors:
                        mut_pt = ptypes.make_mutable(pt)
                        mut_pt.color = norm_colors[sym]

            self._pipeline.modifiers.append(_set_colors)

        # Simulation cell wireframe visibility
        if data.cell is not None and data.cell.vis is not None:
            data.cell.vis.enabled = self.show_cell

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------

    def _make_renderer(self):
        """Return a renderer configured for the active style."""
        r = None
        if self.representation == "vdw-overlay":
            from ovito.vis import OpenGLRenderer
            r = OpenGLRenderer()
            r.order_independent_transparency = True
            
        elif self.style == "hand-drawn":
            from ovito.vis import OSPRayRenderer
            r = OSPRayRenderer()
            r.ambient_brightness = 1.0
            r.direct_light_enabled = False
            r.sky_light_enabled = False
            r.material_shininess = 0
            r.material_specular_brightness = 0.0
            if self.atom_borders:
                r.outlines_enabled = True
                r.outlines_color = (0.0, 0.0, 0.0)

        preset = _STYLE_PRESETS[self.style]
        
        if r is None:
            if self.renderer == "opengl":
                from ovito.vis import OpenGLRenderer
                r = OpenGLRenderer()
            elif self.renderer == "ospray":
                from ovito.vis import OSPRayRenderer
                r = OSPRayRenderer()
                # Try to map some Tachyon preset settings to OSPRay reasonably
                if "direct_light_intensity" in preset:
                    r.direct_light_intensity = preset["direct_light_intensity"]
            else:
                r = TachyonRenderer(**preset)

        # Apply quality settings
        if self.quality == "draft":
            if hasattr(r, "samples_per_pixel"):
                r.samples_per_pixel = 1
                r.denoising_enabled = False
            elif hasattr(r, "antialiasing_samples"):
                r.antialiasing = False
                r.antialiasing_samples = 0
            elif hasattr(r, "antialiasing_level"):
                r.antialiasing_level = 0
        elif self.quality == "medium":
            if hasattr(r, "samples_per_pixel"):
                r.samples_per_pixel = 4
                r.denoising_enabled = True
            elif hasattr(r, "antialiasing_samples"):
                r.antialiasing = True
                r.antialiasing_samples = 4
            elif hasattr(r, "antialiasing_level"):
                r.antialiasing_level = 2
        elif self.quality == "high":
            if hasattr(r, "samples_per_pixel"):
                r.samples_per_pixel = 16
                r.denoising_enabled = True
            elif hasattr(r, "antialiasing_samples"):
                r.antialiasing = True
                r.antialiasing_samples = 12
            elif hasattr(r, "antialiasing_level"):
                r.antialiasing_level = 3
        
        return r

    def _make_viewport(self) -> Viewport:
        """Return a :class:`Viewport` positioned according to camera settings.

        If *camera_distance* is ``None``, falls back to ``zoom_all()`` and then
        re-applies the user's azimuth / elevation while preserving the
        auto-computed distance.
        """
        self._pipeline.add_to_scene()
        data = self._pipeline.compute()

        vp_type = Viewport.Type.Ortho if self.projection == "orthographic" else Viewport.Type.Perspective
        vp = Viewport(type=vp_type)
        vp.zoom_all()

        # Determine focal point (centre of mass if not specified)
        if self.focal_point is not None:
            fp = np.array(self.focal_point, dtype=float)
        else:
            fp = np.mean(data.particles.positions, axis=0)

        # Determine distance
        if self.camera_distance is not None:
            dist = self.camera_distance
            if self.projection == "orthographic":
                vp.fov = dist
        else:
            # Heuristic: 2.5× the bounding-sphere radius
            positions = np.array(data.particles.positions)
            radii = np.linalg.norm(positions - fp, axis=1)
            dist = max(float(np.max(radii)) * 2.5, 5.0)

        # Convert spherical → Cartesian offset from focal point
        az = math.radians(self.camera_azimuth)
        el = math.radians(self.camera_elevation)
        dx = dist * math.cos(el) * math.cos(az)
        dy = dist * math.sin(el)
        dz = dist * math.cos(el) * math.sin(az)
        cam_pos = fp + np.array([dx, dy, dz])
        cam_dir = fp - cam_pos  # look towards focal point

        vp.camera_pos = tuple(cam_pos)
        vp.camera_dir = tuple(cam_dir)

        # Add requested overlays
        if self.show_cell_axes or self.show_info:
            self._add_overlays(vp, data)

        return vp

    def _add_overlays(self, vp: Viewport, data) -> None:
        from ovito.vis import CoordinateTripodOverlay, TextLabelOverlay

        if self.show_cell_axes and data.cell is not None:
            a, b, c = data.cell[:, 0], data.cell[:, 1], data.cell[:, 2]
            # Normalize axes so they are drawn with equal lengths
            a_dir = a / np.linalg.norm(a)
            b_dir = b / np.linalg.norm(b)
            c_dir = c / np.linalg.norm(c)
            
            self._tripod = CoordinateTripodOverlay(
                axis1_dir=tuple(a_dir), axis1_label='a',
                axis2_dir=tuple(b_dir), axis2_label='b',
                axis3_dir=tuple(c_dir), axis3_label='c',
                size=0.1,
                offset_x=0.02,
                offset_y=0.02,
            )
            vp.overlays.append(self._tripod)

        if self.show_info:
            ptypes = data.particles.particle_types
            counts = np.bincount(ptypes)
            formula_parts = []
            for pt in ptypes.types:
                count = counts[pt.id]
                if count > 0:
                    formula_parts.append(f"{pt.name}{count if count > 1 else ''}")
            formula = " ".join(formula_parts)

            bonds_count = data.particles.bonds.count if data.particles.bonds is not None else 0
            lines = [
                f"Formula: {formula}",
                f"Atoms: {data.particles.count}  |  Bonds: {bonds_count}"
            ]

            if data.cell is not None and any(data.cell.pbc):
                a, b, c = data.cell[:, 0], data.cell[:, 1], data.cell[:, 2]
                la = np.linalg.norm(a)
                lb = np.linalg.norm(b)
                lc = np.linalg.norm(c)
                # Angle calculation with clipping to avoid numerical issues
                alpha = np.degrees(np.arccos(np.clip(np.dot(b, c) / (lb * lc), -1.0, 1.0)))
                beta = np.degrees(np.arccos(np.clip(np.dot(a, c) / (la * lc), -1.0, 1.0)))
                gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0)))

                lines.append(f"Cell: a={la:.2f} Å, b={lb:.2f} Å, c={lc:.2f} Å")
                lines.append(f"Angles: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°")

            text = "\n".join(lines)
            vp.overlays.append(TextLabelOverlay(
                text=text,
                font_size=0.03,
                text_color=(1, 1, 1),
                outline_color=(0, 0, 0),
                outline_enabled=True,
            ))

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

        if self._vdw_modifier is not None:
            self._pipeline.modifiers.append(self._vdw_modifier)

        vp = self._make_viewport()
        renderer = self._make_renderer()
        vp.render_image(
            filename=str(output),
            size=(width, height),
            renderer=renderer,
        )
        
        if self._vdw_modifier is not None:
            del self._pipeline.modifiers[-1]
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
        rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
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
            Currently only ``"rotate"`` is supported (360° rotation).
        rotation_axis : tuple[float, float, float]
            The axis of rotation as an (x, y, z) vector.  Default is
            ``(0, 1, 0)`` (Y-axis).  The vector does not need to be
            normalised.

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

        axis = np.array(rotation_axis, dtype=float)
        if np.linalg.norm(axis) == 0:
            raise ValueError("rotation_axis must be a non-zero vector.")
        axis = axis / np.linalg.norm(axis)

        # Add a custom modifier that rotates all particles *and* the
        # simulation cell around the given axis.
        def _rotate(frame: int, data) -> None:  # noqa: ANN001
            import sys
            
            # Print rendering progress
            sys.stdout.write(f"\rRendering frame {frame + 1}/{num_frames}...")
            sys.stdout.flush()
            if frame == num_frames - 1:
                sys.stdout.write("\n")

            angle = 2 * math.pi * frame / num_frames
            rot = _rotation_matrix(axis, angle)

            # Rotate particle positions around their centre of mass
            particles = data.particles_
            positions = particles.positions_
            center = np.mean(positions, axis=0)
            positions[:] = (positions - center) @ rot.T + center

            # Rotate the simulation cell (if present)
            if data.cell is not None:
                cell = data.cell_
                matrix = np.array(cell.matrix)
                cell_vecs = matrix[:3, :3]
                origin = matrix[:3, 3]
                new_vecs = rot @ cell_vecs
                new_origin = rot @ (origin - center) + center
                new_matrix = np.zeros((3, 4))
                new_matrix[:3, :3] = new_vecs
                new_matrix[:3, 3] = new_origin
                cell.matrix = new_matrix

            # Update tripod if it exists (so it rotates with the cell)
            if getattr(self, '_tripod', None) is not None and data.cell is not None:
                new_a = new_vecs[:, 0]
                new_b = new_vecs[:, 1]
                new_c = new_vecs[:, 2]
                self._tripod.axis1_dir = tuple(new_a / np.linalg.norm(new_a))
                self._tripod.axis2_dir = tuple(new_b / np.linalg.norm(new_b))
                self._tripod.axis3_dir = tuple(new_c / np.linalg.norm(new_c))

        self._pipeline.modifiers.append(_rotate)
        if self._vdw_modifier is not None:
            self._pipeline.modifiers.append(self._vdw_modifier)

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
        if self._vdw_modifier is not None:
            del self._pipeline.modifiers[-1]
        del self._pipeline.modifiers[-1]
        self._pipeline.remove_from_scene()
        return output
