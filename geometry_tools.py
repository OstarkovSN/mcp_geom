"""
Core molecular geometry manipulation functions using ASE.
All functions operate on ASE Atoms objects and return modified copies.
"""
import io
import logging
import numpy as np
from ase import Atoms
from ase.io import read, write

logger = logging.getLogger(__name__)


def atoms_from_xyz(xyz_string: str) -> Atoms:
    """Parse an XYZ string into an ASE Atoms object."""
    return read(io.StringIO(xyz_string), format="xyz")


def atoms_to_xyz(atoms: Atoms) -> str:
    """Serialize an ASE Atoms object to an XYZ string."""
    buf = io.StringIO()
    write(buf, atoms, format="xyz")
    return buf.getvalue()


def move_atom(atoms: Atoms, atom_index: int, displacement: list[float]) -> Atoms:
    """
    Move a single atom by a displacement vector.

    Args:
        atoms: ASE Atoms object
        atom_index: 0-based index of the atom to move
        displacement: [dx, dy, dz] in Angstroms

    Returns:
        New Atoms object with the atom moved.
    """
    n = len(atoms)
    if atom_index < 0 or atom_index >= n:
        raise ValueError(f"atom_index {atom_index} out of range [0, {n})")
    result = atoms.copy()
    result.positions[atom_index] += np.array(displacement, dtype=float)
    logger.info("Moved atom %d by %s", atom_index, displacement)
    return result


def move_group(atoms: Atoms, atom_indices: list[int], displacement: list[float]) -> Atoms:
    """
    Move a group of atoms by the same displacement vector.

    Args:
        atoms: ASE Atoms object
        atom_indices: list of 0-based atom indices to move
        displacement: [dx, dy, dz] in Angstroms

    Returns:
        New Atoms object with the group moved.
    """
    n = len(atoms)
    for idx in atom_indices:
        if idx < 0 or idx >= n:
            raise ValueError(f"atom_index {idx} out of range [0, {n})")
    result = atoms.copy()
    disp = np.array(displacement, dtype=float)
    result.positions[atom_indices] += disp
    logger.info("Moved group %s by %s", atom_indices, displacement)
    return result


def set_bond_length(atoms: Atoms, atom_i: int, atom_j: int, new_length: float) -> Atoms:
    """
    Set the distance between two atoms (bond stretching).
    Atom j is moved along the i→j vector; atom i stays fixed.

    Args:
        atoms: ASE Atoms object
        atom_i: index of the fixed atom
        atom_j: index of the atom to move
        new_length: desired bond length in Angstroms

    Returns:
        New Atoms object with adjusted bond length.
    """
    if new_length <= 0:
        raise ValueError(f"new_length must be positive, got {new_length}")
    result = atoms.copy()
    pos_i = result.positions[atom_i]
    pos_j = result.positions[atom_j]
    vec = pos_j - pos_i
    current_length = np.linalg.norm(vec)
    if current_length < 1e-10:
        raise ValueError(f"Atoms {atom_i} and {atom_j} are at the same position")
    new_vec = vec / current_length * new_length
    result.positions[atom_j] = pos_i + new_vec
    logger.info("Set bond length %d-%d to %.4f Å (was %.4f Å)", atom_i, atom_j, new_length, current_length)
    return result


def set_bond_angle(
    atoms: Atoms, atom_i: int, atom_j: int, atom_k: int, new_angle_deg: float
) -> Atoms:
    """
    Set the bond angle i-j-k (vertex at j) to new_angle_deg degrees.
    Atom k is rotated about an axis perpendicular to the i-j-k plane passing through j.
    Atom i and j stay fixed.

    Args:
        atoms: ASE Atoms object
        atom_i, atom_j, atom_k: atom indices defining the angle (vertex at j)
        new_angle_deg: desired angle in degrees

    Returns:
        New Atoms object with adjusted angle.
    """
    result = atoms.copy()
    pos_i = result.positions[atom_i]
    pos_j = result.positions[atom_j]
    pos_k = result.positions[atom_k]

    vec_ji = pos_i - pos_j
    vec_jk = pos_k - pos_j

    norm_ji = np.linalg.norm(vec_ji)
    norm_jk = np.linalg.norm(vec_jk)
    if norm_ji < 1e-10 or norm_jk < 1e-10:
        raise ValueError(f"Degenerate bond vectors for angle {atom_i}-{atom_j}-{atom_k}")

    vec_ji_n = vec_ji / norm_ji
    vec_jk_n = vec_jk / norm_jk

    cos_cur = np.clip(np.dot(vec_ji_n, vec_jk_n), -1.0, 1.0)
    current_angle = np.degrees(np.arccos(cos_cur))

    # Rotation axis: perpendicular to the plane containing i-j-k
    axis = np.cross(vec_ji_n, vec_jk_n)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        # Atoms are collinear; pick arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(vec_ji_n[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(vec_ji_n, perp)
        axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm

    delta_angle = np.radians(new_angle_deg - current_angle)
    # Rotate vec_jk around axis by delta_angle using Rodrigues' formula
    new_vec_jk = _rodrigues(vec_jk, axis, delta_angle)
    result.positions[atom_k] = pos_j + new_vec_jk
    logger.info(
        "Set angle %d-%d-%d to %.2f° (was %.2f°)", atom_i, atom_j, atom_k, new_angle_deg, current_angle
    )
    return result


def set_dihedral(
    atoms: Atoms, atom_i: int, atom_j: int, atom_k: int, atom_l: int, new_dihedral_deg: float
) -> Atoms:
    """
    Set the dihedral angle i-j-k-l to new_dihedral_deg degrees.
    Rotates atom l (and no other atoms) about the j-k bond axis.

    Args:
        atoms: ASE Atoms object
        atom_i, atom_j, atom_k, atom_l: atoms defining the dihedral
        new_dihedral_deg: desired dihedral in degrees (-180 to 180)

    Returns:
        New Atoms object with adjusted dihedral.
    """
    result = atoms.copy()
    pos_i = result.positions[atom_i]
    pos_j = result.positions[atom_j]
    pos_k = result.positions[atom_k]
    pos_l = result.positions[atom_l]

    b1 = pos_j - pos_i
    b2 = pos_k - pos_j
    b3 = pos_l - pos_k

    b2_norm = np.linalg.norm(b2)
    if b2_norm < 1e-10:
        raise ValueError(f"Atoms {atom_j} and {atom_k} are at the same position")
    b2_n = b2 / b2_norm

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        raise ValueError("Degenerate dihedral (collinear atoms)")

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_d = np.clip(np.dot(n1, n2), -1.0, 1.0)
    sin_d = np.dot(np.cross(n1, n2), b2_n)
    current_dihedral = np.degrees(np.arctan2(sin_d, cos_d))

    delta = new_dihedral_deg - current_dihedral
    # Rotate atom_l about the j-k axis (b2_n) passing through k
    result.positions[atom_l] = pos_k + _rodrigues(b3, b2_n, np.radians(delta))
    logger.info(
        "Set dihedral %d-%d-%d-%d to %.2f° (was %.2f°)",
        atom_i, atom_j, atom_k, atom_l, new_dihedral_deg, current_dihedral,
    )
    return result


def get_bond_length(atoms: Atoms, atom_i: int, atom_j: int) -> float:
    """Return the distance between two atoms in Angstroms."""
    return float(np.linalg.norm(atoms.positions[atom_j] - atoms.positions[atom_i]))


def get_bond_angle(atoms: Atoms, atom_i: int, atom_j: int, atom_k: int) -> float:
    """Return the angle i-j-k in degrees (vertex at j)."""
    pos_i = atoms.positions[atom_i]
    pos_j = atoms.positions[atom_j]
    pos_k = atoms.positions[atom_k]
    vec_ji = pos_i - pos_j
    vec_jk = pos_k - pos_j
    norm_ji = np.linalg.norm(vec_ji)
    norm_jk = np.linalg.norm(vec_jk)
    if norm_ji < 1e-10 or norm_jk < 1e-10:
        raise ValueError(f"Degenerate bond vectors for angle {atom_i}-{atom_j}-{atom_k}")
    cos_a = np.dot(vec_ji, vec_jk) / (norm_ji * norm_jk)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def get_dihedral(atoms: Atoms, atom_i: int, atom_j: int, atom_k: int, atom_l: int) -> float:
    """Return the dihedral angle i-j-k-l in degrees."""
    pos_i = atoms.positions[atom_i]
    pos_j = atoms.positions[atom_j]
    pos_k = atoms.positions[atom_k]
    pos_l = atoms.positions[atom_l]
    b1 = pos_j - pos_i
    b2 = pos_k - pos_j
    b3 = pos_l - pos_k
    b2_norm = np.linalg.norm(b2)
    if b2_norm < 1e-10:
        raise ValueError(f"Atoms {atom_j} and {atom_k} are at the same position")
    b2_n = b2 / b2_norm
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        raise ValueError("Degenerate dihedral (collinear atoms)")
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    cos_d = np.clip(np.dot(n1, n2), -1.0, 1.0)
    sin_d = np.dot(np.cross(n1, n2), b2_n)
    return float(np.degrees(np.arctan2(sin_d, cos_d)))


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector v by angle_rad radians around unit axis using Rodrigues' formula."""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


def detect_fragment(atoms: Atoms, atom_j: int, atom_k: int, start_atom: int) -> list[int]:
    """
    Given bond j-k, find all atoms reachable from start_atom WITHOUT crossing the j-k bond.
    Uses BFS with cutoff-based neighbor detection.

    Args:
        atoms: ASE Atoms object
        atom_j: first atom of the bond that acts as a barrier
        atom_k: second atom of the bond that acts as a barrier
        start_atom: atom to start BFS from (must be on the side of the fragment)

    Returns:
        Sorted list of atom indices in the fragment (including start_atom).
    """
    from collections import deque
    from ase.neighborlist import neighbor_list
    from ase.data import covalent_radii

    atomic_numbers = atoms.get_atomic_numbers()
    n = len(atoms)

    # Build per-element cutoff: max radius seen in the molecule, so cutoff = 1.2 * (r_i + r_j)
    # Using the largest radius gives the uniform bound without O(n²) pair enumeration.
    radii = covalent_radii[atomic_numbers]  # shape (n,)
    max_radius = radii.max()
    uniform_cutoff = 1.2 * 2 * max_radius

    # Request distances 'd' from neighbor_list to avoid recomputing norms per pair
    i_idx, j_idx, dists = neighbor_list('ijd', atoms, uniform_cutoff)

    # Build adjacency, keeping only pairs within their specific per-pair cutoff
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for ii, jj, dist in zip(i_idx, j_idx, dists):
        if dist <= 1.2 * (radii[ii] + radii[jj]):
            adjacency[ii].append(jj)

    # BFS from start_atom, not crossing the j-k bond in either direction
    visited = set()
    queue: deque[int] = deque([start_atom])
    visited.add(start_atom)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            if (current == atom_j and neighbor == atom_k) or (current == atom_k and neighbor == atom_j):
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    fragment = sorted(visited)
    logger.info(
        "Detected fragment from atom %d avoiding bond %d-%d: %s",
        start_atom, atom_j, atom_k, fragment,
    )
    return fragment


def rotate_dihedral_fragment(
    atoms: Atoms, atom_j: int, atom_k: int, delta_deg: float, fragment_indices: list[int]
) -> Atoms:
    """
    Rotate all atoms in fragment_indices around the j-k bond axis by delta_deg degrees.
    The rotation axis passes through atom_k.

    Args:
        atoms: ASE Atoms object
        atom_j: first atom defining the rotation axis
        atom_k: second atom defining the rotation axis (pivot point)
        delta_deg: rotation angle in degrees
        fragment_indices: indices of atoms to rotate rigidly

    Returns:
        New Atoms object with rotated fragment.
    """
    result = atoms.copy()
    pos_j = result.positions[atom_j]
    pos_k = result.positions[atom_k]

    axis = pos_k - pos_j
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        raise ValueError(f"Atoms {atom_j} and {atom_k} are at the same position")
    axis = axis / axis_norm

    delta_rad = np.radians(delta_deg)
    for idx in fragment_indices:
        vec = result.positions[idx] - pos_k
        rotated = _rodrigues(vec, axis, delta_rad)
        result.positions[idx] = pos_k + rotated

    logger.info(
        "Rotated fragment %s around bond %d-%d by %.2f°",
        fragment_indices, atom_j, atom_k, delta_deg,
    )
    return result


def set_bond_angle_fragment(
    atoms: Atoms,
    atom_i: int,
    atom_j: int,
    atom_k: int,
    new_angle_deg: float,
    fragment_indices: list[int],
) -> Atoms:
    """
    Set the bond angle i-j-k (vertex at j) to new_angle_deg by rotating ALL atoms in
    fragment_indices rigidly around the axis perpendicular to the i-j-k plane, pivoting
    at atom_j. fragment_indices should be all atoms on the atom_k side of the j-k bond.

    Args:
        atoms: ASE Atoms object
        atom_i, atom_j, atom_k: atom indices defining the angle (vertex at j)
        new_angle_deg: desired angle in degrees
        fragment_indices: indices of atoms to rotate together

    Returns:
        New Atoms object with adjusted angle.
    """
    result = atoms.copy()
    pos_i = result.positions[atom_i]
    pos_j = result.positions[atom_j]
    pos_k = result.positions[atom_k]

    vec_ji = pos_i - pos_j
    vec_jk = pos_k - pos_j

    norm_ji = np.linalg.norm(vec_ji)
    norm_jk = np.linalg.norm(vec_jk)
    if norm_ji < 1e-10 or norm_jk < 1e-10:
        raise ValueError(f"Degenerate bond vectors for angle {atom_i}-{atom_j}-{atom_k}")

    vec_ji_n = vec_ji / norm_ji
    vec_jk_n = vec_jk / norm_jk

    current_angle = get_bond_angle(atoms, atom_i, atom_j, atom_k)

    # Rotation axis: perpendicular to the plane containing i-j-k
    axis = np.cross(vec_ji_n, vec_jk_n)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        # Atoms are collinear; pick arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(vec_ji_n[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(vec_ji_n, perp)
        axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm

    delta_angle = np.radians(new_angle_deg - current_angle)

    for idx in fragment_indices:
        vec = result.positions[idx] - pos_j
        rotated = _rodrigues(vec, axis, delta_angle)
        result.positions[idx] = pos_j + rotated

    logger.info(
        "Set angle (fragment) %d-%d-%d to %.2f° (was %.2f°), rotated fragment %s",
        atom_i, atom_j, atom_k, new_angle_deg, current_angle, fragment_indices,
    )
    return result


def set_dihedral_fragment(
    atoms: Atoms,
    atom_i: int,
    atom_j: int,
    atom_k: int,
    atom_l: int,
    new_dihedral_deg: float,
    fragment_indices: list[int],
) -> Atoms:
    """
    Set the dihedral angle i-j-k-l to new_dihedral_deg by rotating ALL atoms in
    fragment_indices rigidly around the j-k axis. fragment_indices should be all
    atoms on the l-side of the j-k bond.

    Args:
        atoms: ASE Atoms object
        atom_i, atom_j, atom_k, atom_l: atoms defining the dihedral
        new_dihedral_deg: desired dihedral in degrees
        fragment_indices: indices of atoms to rotate together

    Returns:
        New Atoms object with adjusted dihedral.
    """
    current_dihedral = get_dihedral(atoms, atom_i, atom_j, atom_k, atom_l)
    delta = new_dihedral_deg - current_dihedral

    result = rotate_dihedral_fragment(atoms, atom_j, atom_k, delta, fragment_indices)
    logger.info(
        "Set dihedral (fragment) %d-%d-%d-%d to %.2f° (was %.2f°)",
        atom_i, atom_j, atom_k, atom_l, new_dihedral_deg, current_dihedral,
    )
    return result


def get_center_of_mass(atoms: Atoms) -> np.ndarray:
    """
    Return the center of mass of the molecule as a numpy array.

    Args:
        atoms: ASE Atoms object

    Returns:
        Center of mass coordinates as np.ndarray of shape (3,) in Angstroms.
    """
    return atoms.get_center_of_mass()


def translate_to_origin(atoms: Atoms) -> Atoms:
    """
    Return a new Atoms object with the center of mass moved to the origin.

    Args:
        atoms: ASE Atoms object

    Returns:
        New Atoms object centered at [0, 0, 0].
    """
    result = atoms.copy()
    result.positions -= get_center_of_mass(atoms)
    return result


def rotate_molecule(atoms: Atoms, axis: list[float], angle_deg: float) -> Atoms:
    """
    Rotate the entire molecule around a given axis passing through the center of mass.

    Args:
        atoms: ASE Atoms object
        axis: list[float] of length 3 — rotation axis vector (will be normalized)
        angle_deg: rotation angle in degrees

    Returns:
        New Atoms object with all atoms rotated.
    """
    result = atoms.copy()
    com = get_center_of_mass(atoms)
    ax = np.array(axis, dtype=float)
    ax_norm = np.linalg.norm(ax)
    if ax_norm < 1e-10:
        raise ValueError("Rotation axis cannot be the zero vector")
    ax = ax / ax_norm
    angle_rad = np.radians(angle_deg)
    for i in range(len(result)):
        vec = result.positions[i] - com
        result.positions[i] = com + _rodrigues(vec, ax, angle_rad)
    logger.info("Rotated molecule by %.2f° around axis %s", angle_deg, axis)
    return result
