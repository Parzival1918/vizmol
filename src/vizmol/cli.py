"""Command-line interface for vizmol.

Usage examples::

    vizmol render structure.xyz -o image.png --style realistic
    vizmol animate structure.cif -o movie.avi --frames 120 --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vizmol.core import MoleculeVisualizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vizmol",
        description="Render images and animations of molecules and crystals.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- shared arguments ------------------------------------------------
    def _add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "input",
            type=str,
            help="Path to an atomic-structure file (XYZ, CIF, PDB, POSCAR, …).",
        )
        sub.add_argument(
            "-o", "--output",
            type=str,
            required=True,
            help="Output file path.",
        )
        sub.add_argument(
            "--style",
            choices=["realistic", "cartoon"],
            default="realistic",
            help="Rendering style (default: realistic).",
        )
        sub.add_argument(
            "--representation",
            choices=["ball-and-stick", "space-filling", "wireframe"],
            default="ball-and-stick",
            help="Atom/bond representation (default: ball-and-stick).",
        )
        sub.add_argument(
            "--bond-padding",
            type=float,
            default=0.3,
            help="Padding (Å) added to the sum of covalent radii for bond "
                 "detection (default: 0.3).",
        )
        sub.add_argument(
            "--supercell",
            type=str,
            default=None,
            metavar="NxNxN",
            help="Replicate the unit cell, e.g. '2x2x2' for a 2×2×2 "
                 "supercell.",
        )
        cell_group = sub.add_mutually_exclusive_group()
        cell_group.add_argument(
            "--show-cell",
            action="store_true",
            default=None,
            help="Show the simulation-cell wireframe.",
        )
        cell_group.add_argument(
            "--hide-cell",
            action="store_true",
            default=None,
            help="Hide the simulation-cell wireframe.",
        )
        sub.add_argument(
            "--width",
            type=int,
            default=800,
            help="Image width in pixels (default: 800).",
        )
        sub.add_argument(
            "--height",
            type=int,
            default=600,
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
        default=60,
        help="Number of frames in the animation (default: 60).",
    )
    animate_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30).",
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


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vizmol`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

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
        )
        print(f"Animation saved to {out}")


if __name__ == "__main__":
    main()
