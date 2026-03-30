"""
MCP server for molecular geometry editing.
Run as: uv run python mcp_server.py
Communicates via stdio (default MCP transport).
"""
import logging
import sys
from mcp.server.fastmcp import FastMCP
import os
from geometry_tools import (
    atoms_from_xyz,
    atoms_to_xyz,
    move_atom,
    move_group,
    set_bond_length,
    set_bond_angle,
    set_dihedral,
    get_bond_length,
    get_bond_angle,
    get_dihedral,
    detect_fragment,
    set_dihedral_fragment,
)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-geom", instructions="Molecular geometry editing tools")

# Server-side state: current molecule as XYZ string
_current_xyz: str | None = None


def _require_molecule() -> str:
    if _current_xyz is None:
        raise ValueError("No molecule loaded. Call load_molecule first.")
    return _current_xyz


# ─────────────────────────────────────────────
# Tool: load / get molecule
# ─────────────────────────────────────────────

@mcp.tool()
def load_molecule(xyz_string: str) -> str:
    """
    Load a molecule from an XYZ-format string.
    Returns a summary of the loaded molecule (atom count and formula).
    The molecule is stored as server state for subsequent editing tools.
    """
    global _current_xyz
    atoms = atoms_from_xyz(xyz_string)
    _current_xyz = atoms_to_xyz(atoms)
    formula = atoms.get_chemical_formula()
    return f"Loaded molecule: {formula} ({len(atoms)} atoms)"


@mcp.tool()
def get_molecule() -> str:
    """
    Return the current molecule in XYZ format.
    """
    return _require_molecule()


@mcp.tool()
def get_atom_info() -> str:
    """
    Return a table of atoms with their indices, symbols, and coordinates.
    Useful to identify which atom index to use in other tools.
    """
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    lines = ["idx | sym |      x      |      y      |      z"]
    lines.append("-" * 55)
    for i, (sym, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
        lines.append(f"{i:3d} | {sym:3s} | {pos[0]:11.6f} | {pos[1]:11.6f} | {pos[2]:11.6f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Tool: move single atom
# ─────────────────────────────────────────────

@mcp.tool()
def move_single_atom(atom_index: int, dx: float, dy: float, dz: float) -> str:
    """
    Move a single atom by a displacement vector (dx, dy, dz) in Angstroms.

    Args:
        atom_index: 0-based index of the atom to move
        dx: displacement in x direction (Angstroms)
        dy: displacement in y direction (Angstroms)
        dz: displacement in z direction (Angstroms)

    Returns:
        Updated molecule in XYZ format.
    """
    global _current_xyz
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    atoms = move_atom(atoms, atom_index, [dx, dy, dz])
    _current_xyz = atoms_to_xyz(atoms)
    sym = atoms.get_chemical_symbols()[atom_index]
    return f"Moved atom {atom_index} ({sym}) by ({dx:.3f}, {dy:.3f}, {dz:.3f}) Å\n\n{_current_xyz}"


# ─────────────────────────────────────────────
# Tool: move group of atoms
# ─────────────────────────────────────────────

@mcp.tool()
def move_atom_group(atom_indices: list[int], dx: float, dy: float, dz: float) -> str:
    """
    Move a group of atoms by a displacement vector (dx, dy, dz) in Angstroms.

    Args:
        atom_indices: list of 0-based atom indices to move together
        dx: displacement in x direction (Angstroms)
        dy: displacement in y direction (Angstroms)
        dz: displacement in z direction (Angstroms)

    Returns:
        Updated molecule in XYZ format.
    """
    global _current_xyz
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    atoms = move_group(atoms, atom_indices, [dx, dy, dz])
    _current_xyz = atoms_to_xyz(atoms)
    return f"Moved atoms {atom_indices} by ({dx:.3f}, {dy:.3f}, {dz:.3f}) Å\n\n{_current_xyz}"


# ─────────────────────────────────────────────
# Tool: bond length
# ─────────────────────────────────────────────

@mcp.tool()
def measure_bond_length(atom_i: int, atom_j: int) -> str:
    """
    Measure the distance between two atoms.

    Args:
        atom_i: index of first atom
        atom_j: index of second atom

    Returns:
        Distance in Angstroms.
    """
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    d = get_bond_length(atoms, atom_i, atom_j)
    syms = atoms.get_chemical_symbols()
    return f"Distance {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]}): {d:.6f} Å"


@mcp.tool()
def change_bond_length(atom_i: int, atom_j: int, new_length: float) -> str:
    """
    Set the bond length between atom_i and atom_j to new_length Angstroms.
    atom_i is held fixed; atom_j is moved along the bond axis.

    Args:
        atom_i: index of the fixed atom
        atom_j: index of the atom to move
        new_length: desired bond length in Angstroms

    Returns:
        Updated molecule in XYZ format.
    """
    global _current_xyz
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    old_length = get_bond_length(atoms, atom_i, atom_j)
    atoms = set_bond_length(atoms, atom_i, atom_j, new_length)
    _current_xyz = atoms_to_xyz(atoms)
    syms = atoms.get_chemical_symbols()
    return (
        f"Bond {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]}): "
        f"{old_length:.4f} Å → {new_length:.4f} Å\n\n{_current_xyz}"
    )


# ─────────────────────────────────────────────
# Tool: bond angle
# ─────────────────────────────────────────────

@mcp.tool()
def measure_bond_angle(atom_i: int, atom_j: int, atom_k: int) -> str:
    """
    Measure the bond angle i-j-k (vertex at j).

    Args:
        atom_i: index of first atom
        atom_j: index of vertex atom
        atom_k: index of third atom

    Returns:
        Angle in degrees.
    """
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    angle = get_bond_angle(atoms, atom_i, atom_j, atom_k)
    syms = atoms.get_chemical_symbols()
    return (
        f"Angle {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]})-{atom_k}({syms[atom_k]}): "
        f"{angle:.4f}°"
    )


@mcp.tool()
def change_bond_angle(atom_i: int, atom_j: int, atom_k: int, new_angle_deg: float) -> str:
    """
    Set the bond angle i-j-k (vertex at j) to new_angle_deg degrees.
    atom_k is rotated; atom_i and atom_j stay fixed.

    Args:
        atom_i: index of first atom
        atom_j: index of the vertex atom (stays fixed)
        atom_k: index of the atom to move
        new_angle_deg: desired angle in degrees

    Returns:
        Updated molecule in XYZ format.
    """
    global _current_xyz
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    old_angle = get_bond_angle(atoms, atom_i, atom_j, atom_k)
    atoms = set_bond_angle(atoms, atom_i, atom_j, atom_k, new_angle_deg)
    _current_xyz = atoms_to_xyz(atoms)
    syms = atoms.get_chemical_symbols()
    return (
        f"Angle {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]})-{atom_k}({syms[atom_k]}): "
        f"{old_angle:.2f}° → {new_angle_deg:.2f}°\n\n{_current_xyz}"
    )


# ─────────────────────────────────────────────
# Tool: dihedral angle
# ─────────────────────────────────────────────

@mcp.tool()
def measure_dihedral_angle(atom_i: int, atom_j: int, atom_k: int, atom_l: int) -> str:
    """
    Measure the dihedral angle i-j-k-l.

    Args:
        atom_i, atom_j, atom_k, atom_l: four atom indices defining the dihedral

    Returns:
        Dihedral angle in degrees (-180 to 180).
    """
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    d = get_dihedral(atoms, atom_i, atom_j, atom_k, atom_l)
    syms = atoms.get_chemical_symbols()
    return (
        f"Dihedral {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]})-"
        f"{atom_k}({syms[atom_k]})-{atom_l}({syms[atom_l]}): {d:.4f}°"
    )


@mcp.tool()
def change_dihedral_angle(
    atom_i: int, atom_j: int, atom_k: int, atom_l: int, new_dihedral_deg: float
) -> str:
    """
    Set the dihedral angle i-j-k-l to new_dihedral_deg degrees.
    Rotates atom_l about the j-k bond axis.

    Args:
        atom_i, atom_j, atom_k, atom_l: four atom indices defining the dihedral
        new_dihedral_deg: desired dihedral in degrees (-180 to 180)

    Returns:
        Updated molecule in XYZ format.
    """
    global _current_xyz
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    old_d = get_dihedral(atoms, atom_i, atom_j, atom_k, atom_l)
    atoms = set_dihedral(atoms, atom_i, atom_j, atom_k, atom_l, new_dihedral_deg)
    _current_xyz = atoms_to_xyz(atoms)
    syms = atoms.get_chemical_symbols()
    return (
        f"Dihedral {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]})-"
        f"{atom_k}({syms[atom_k]})-{atom_l}({syms[atom_l]}): "
        f"{old_d:.2f}° → {new_dihedral_deg:.2f}°\n\n{_current_xyz}"
    )


# ─────────────────────────────────────────────
# Tool: fragment-aware dihedral angle
# ─────────────────────────────────────────────

@mcp.tool()
def change_dihedral_angle_fragment(
    atom_i: int, atom_j: int, atom_k: int, atom_l: int, new_dihedral_deg: float
) -> str:
    """
    Set the dihedral angle i-j-k-l to new_dihedral_deg degrees by rotating the ENTIRE
    fragment on the atom_l side of the j-k bond. This preserves the internal geometry
    of that fragment (all atoms move together rigidly).

    The fragment is auto-detected: all atoms reachable from atom_l without crossing
    the j-k bond.

    Args:
        atom_i, atom_j, atom_k, atom_l: four atom indices defining the dihedral
        new_dihedral_deg: desired dihedral in degrees (-180 to 180)

    Returns:
        Updated molecule in XYZ format.
    """
    global _current_xyz
    xyz = _require_molecule()
    atoms = atoms_from_xyz(xyz)
    old_d = get_dihedral(atoms, atom_i, atom_j, atom_k, atom_l)
    fragment = detect_fragment(atoms, atom_j, atom_k, atom_l)
    atoms = set_dihedral_fragment(atoms, atom_i, atom_j, atom_k, atom_l, new_dihedral_deg, fragment)
    _current_xyz = atoms_to_xyz(atoms)
    syms = atoms.get_chemical_symbols()
    return (
        f"Dihedral (fragment) {atom_i}({syms[atom_i]})-{atom_j}({syms[atom_j]})-"
        f"{atom_k}({syms[atom_k]})-{atom_l}({syms[atom_l]}): "
        f"{old_d:.2f}° → {new_dihedral_deg:.2f}° "
        f"(rotated fragment: {fragment})\n\n{_current_xyz}"
    )


# ─────────────────────────────────────────────
# Tool: save molecule
# ─────────────────────────────────────────────

@mcp.tool()
def save_molecule(filename: str) -> str:
    """
    Save the current molecule to a file in XYZ format.
    The filename is relative to the server's working directory.

    Args:
        filename: path to write the molecule to (e.g. "output.xyz")

    Returns:
        Confirmation message with the absolute path of the saved file.
    """
    xyz = _require_molecule()
    abs_path = os.path.abspath(filename)
    with open(abs_path, "w") as fh:
        fh.write(xyz)
    logger.info("Saved molecule to %s", abs_path)
    return f"Molecule saved to: {abs_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
