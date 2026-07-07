"""Command-line interface for vizmol.

Usage examples::

    vizmol render structure.xyz -o image.png --style realistic
    vizmol animate structure.cif -o movie.avi --frames 120 --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    pass  # python < 3.11, though pyproject.toml requires >= 3.11

from vizmol.core import MoleculeVisualizer


def _build_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vizmol",
        description="Render images and animations of molecules and crystals.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to a TOML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- shared arguments ------------------------------------------------
    def _add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "input",
            type=str,
            nargs="?",
            default=config.get("input"),
            help="Path to an atomic-structure file (XYZ, CIF, PDB, POSCAR, …).",
        )
        sub.add_argument(
            "-o", "--output",
            type=str,
            default=config.get("output"),
            help="Output file path.",
        )
        sub.add_argument(
            "--style",
            choices=["realistic", "cartoon", "flat", "hand-drawn"],
            default=config.get("style", "realistic"),
            help="Rendering style (default: realistic).",
        )
        sub.add_argument(
            "--renderer",
            choices=["tachyon", "ospray", "opengl"],
            default=config.get("renderer", None),
            help="Explicitly set the rendering engine. If unset, it is chosen automatically.",
        )
        sub.add_argument(
            "--quality",
            choices=["draft", "medium", "high"],
            default=config.get("quality", "high"),
            help="Rendering quality level (default: high).",
        )
        sub.add_argument(
            "--color-scheme",
            type=str,
            default=config.get("color_scheme", "default"),
            help="Preset color scheme (e.g. default, muted, xyzrender, pmol, paton).",
        )
        sub.add_argument(
            "--representation",
            choices=[
                "ball-and-stick", "space-filling", "wireframe", "uniform",
                "tube", "wire", "vdw", "pmol", "paton", "skeletal", "vdw-overlay"
            ],
            default=config.get("representation", "ball-and-stick"),
            help="Atom/bond representation (default: ball-and-stick).",
        )
        sub.add_argument(
            "--bond-padding",
            type=float,
            default=config.get("bond_padding", 0.3),
            help="Padding (Å) added to the sum of covalent radii for bond "
                 "detection (default: 0.3).",
        )
        sub.add_argument(
            "--supercell",
            type=str,
            default=config.get("supercell"),
            metavar="NxNxN",
            help="Replicate the unit cell, e.g. '2x2x2' for a 2×2×2 "
                 "supercell.",
        )
        cell_group = sub.add_mutually_exclusive_group()
        cell_group.add_argument(
            "--show-cell",
            action="store_true",
            default=config.get("show_cell"),
            help="Show the simulation-cell wireframe.",
        )
        cell_group.add_argument(
            "--hide-cell",
            action="store_true",
            default=config.get("hide_cell"),
            help="Hide the simulation-cell wireframe.",
        )
        sub.add_argument(
            "--camera-azimuth",
            type=float,
            default=config.get("camera_azimuth", 45.0),
            help="Horizontal camera angle in degrees (default: 45.0).",
        )
        sub.add_argument(
            "--camera-elevation",
            type=float,
            default=config.get("camera_elevation", 30.0),
            help="Vertical camera angle in degrees (default: 30.0).",
        )
        sub.add_argument(
            "--camera-distance",
            type=float,
            default=config.get("camera_distance"),
            help="Camera distance in Å. If not given, auto-computes a distance.",
        )
        sub.add_argument(
            "--show-cell-axes",
            action="store_true",
            default=config.get("show_cell_axes", False),
            help="Show the unit cell vectors (a, b, c) as a coordinate tripod.",
        )
        sub.add_argument(
            "--show-info",
            action="store_true",
            default=config.get("show_info", False),
            help="Show an overlay with crystal/molecular information.",
        )
        sub.add_argument(
            "--hide-hc-hydrogens",
            action="store_true",
            default=config.get("hide_hc_hydrogens", False),
            help="Remove hydrogen atoms that are bonded to carbon atoms.",
        )
        sub.add_argument(
            "--atom-borders",
            action="store_true",
            default=config.get("atom_borders", False),
            help="Show black borders around atoms (only applies to 'hand-drawn' style).",
        )
        sub.add_argument(
            "--focal-point",
            type=str,
            default=config.get("focal_point"),
            metavar="X,Y,Z",
            help="The 3-D point the camera looks at. Defaults to centre of mass.",
        )
        sub.add_argument(
            "--projection",
            choices=["perspective", "orthographic"],
            default=config.get("projection", "orthographic"),
            help="Camera projection type (default: orthographic).",
        )
        sub.add_argument(
            "--width",
            type=int,
            default=config.get("width", 800),
            help="Image width in pixels (default: 800).",
        )
        sub.add_argument(
            "--height",
            type=int,
            default=config.get("height", 600),
            help="Image height in pixels (default: 600).",
        )

    # ---- render sub-command ----------------------------------------------
    render_parser = subparsers.add_parser(
        "render",
        help="Render a static image.",
    )
    _add_common(render_parser)

    # ---- animate sub-command ---------------------------------------------
    animate_parser = subparsers.add_parser(
        "animate",
        help="Render a camera-rotation animation.",
    )
    _add_common(animate_parser)
    animate_parser.add_argument(
        "--frames",
        type=int,
        default=config.get("frames", 60),
        help="Number of frames in the animation (default: 60).",
    )
    animate_parser.add_argument(
        "--fps",
        type=int,
        default=config.get("fps", 30),
        help="Frames per second (default: 30).",
    )
    animate_parser.add_argument(
        "--rotation-axis",
        type=str,
        default=config.get("rotation_axis", "0,1,0"),
        metavar="X,Y,Z",
        help="Axis of rotation for the animation (default: 0,1,0).",
    )

    return parser


def _parse_supercell(value: str | None) -> tuple[int, int, int] | None:
    """Parse a '2x2x2'-style supercell string into a tuple."""
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid supercell format: {value!r}. Expected NxNxN, e.g. '2x2x2'."
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _parse_vector(value: str | None) -> tuple[float, float, float] | None:
    """Parse a 'X,Y,Z'-style string into a tuple of floats."""
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid vector format: {value!r}. Expected X,Y,Z, e.g. '0,1,0'."
        )
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vizmol`` CLI."""
    if argv is None:
        argv = sys.argv[1:]

    # Pre-parse toml config if provided
    config = {}
    for i, arg in enumerate(argv):
        if arg in ("-c", "--config") and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            if config_path.exists():
                with open(config_path, "rb") as f:
                    raw_config = tomllib.load(f)
                    # Normalize hyphens to underscores for argparse defaults
                    config = {k.replace("-", "_"): v for k, v in raw_config.items()}
            break

    # If config specifies a command but CLI does not, inject it so argparse
    # evaluates the subparser correctly.
    cmd_from_config = config.get("command")
    if cmd_from_config in ("render", "animate"):
        if "render" not in argv and "animate" not in argv:
            argv.append(cmd_from_config)

    parser = _build_parser(config)
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(1)

    if not getattr(args, "input", None):
        parser.error("The input file is required via CLI or config file.")
    if not getattr(args, "output", None):
        parser.error("The output file (-o/--output) is required via CLI or config file.")

    # Resolve show_cell: None means auto-detect
    if args.show_cell:
        show_cell = True
    elif args.hide_cell:
        show_cell = False
    else:
        show_cell = None

    viz = MoleculeVisualizer(
        file_path=args.input,
        bond_padding=args.bond_padding,
        style=args.style,
        representation=args.representation,
        supercell=_parse_supercell(args.supercell),
        show_cell=show_cell,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_distance=args.camera_distance,
        focal_point=_parse_vector(args.focal_point),
        projection=args.projection,
        color_scheme=args.color_scheme,
        colors=config.get("colors"),
        show_cell_axes=args.show_cell_axes,
        show_info=args.show_info,
        hide_hc_hydrogens=args.hide_hc_hydrogens,
        atom_borders=args.atom_borders,
        renderer=args.renderer,
        quality=args.quality,
    )

    print(
        f"Loaded {viz.num_particles} atoms from {args.input} "
        f"({viz.num_bonds} bonds detected)"
    )

    if args.command == "render":
        out = viz.render_image(
            args.output,
            width=args.width,
            height=args.height,
        )
        print(f"Image saved to {out}")

    elif args.command == "animate":
        out = viz.render_animation(
            args.output,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            fps=args.fps,
            rotation_axis=_parse_vector(args.rotation_axis),
        )
        print(f"Animation saved to {out}")


if __name__ == "__main__":
    main()
