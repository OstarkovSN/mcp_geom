"""
Unit tests for geometry_tools.py
"""
import numpy as np
import pytest
from ase import Atoms

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
    rotate_dihedral_fragment,
    set_dihedral_fragment,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

ETHANE_XYZ = """\
8
Ethane
C   0.000000   0.000000   0.000000
C   1.540000   0.000000   0.000000
H  -0.390000   1.027000   0.000000
H  -0.390000  -0.513500  -0.889500
H  -0.390000  -0.513500   0.889500
H   1.930000   1.027000   0.000000
H   1.930000  -0.513500  -0.889500
H   1.930000  -0.513500   0.889500
"""

WATER_XYZ = """\
3
Water
O   0.000000   0.000000   0.119748
H   0.000000   0.756950  -0.478993
H   0.000000  -0.756950  -0.478993
"""


@pytest.fixture
def ethane():
    return atoms_from_xyz(ETHANE_XYZ)


@pytest.fixture
def water():
    return atoms_from_xyz(WATER_XYZ)


# ─────────────────────────────────────────────
# XYZ round-trip
# ─────────────────────────────────────────────

def test_xyz_roundtrip():
    atoms = atoms_from_xyz(ETHANE_XYZ)
    xyz = atoms_to_xyz(atoms)
    atoms2 = atoms_from_xyz(xyz)
    assert len(atoms2) == len(atoms)
    np.testing.assert_allclose(atoms2.positions, atoms.positions, atol=1e-5)


def test_xyz_formula(ethane):
    assert ethane.get_chemical_formula() == "C2H6"


# ─────────────────────────────────────────────
# move_atom
# ─────────────────────────────────────────────

def test_move_atom_single(ethane):
    original_pos = ethane.positions[0].copy()
    result = move_atom(ethane, 0, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.positions[0], original_pos + [1.0, 0.0, 0.0], atol=1e-10)
    # Other atoms unchanged
    np.testing.assert_allclose(result.positions[1:], ethane.positions[1:], atol=1e-10)


def test_move_atom_does_not_mutate(ethane):
    original = ethane.positions.copy()
    move_atom(ethane, 0, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ethane.positions, original)


def test_move_atom_out_of_range(ethane):
    with pytest.raises(ValueError):
        move_atom(ethane, 100, [1.0, 0.0, 0.0])


# ─────────────────────────────────────────────
# move_group
# ─────────────────────────────────────────────

def test_move_group(ethane):
    indices = [0, 2, 3, 4]  # C1 and its H atoms
    disp = [0.0, 0.0, 1.0]
    original = ethane.positions.copy()
    result = move_group(ethane, indices, disp)
    for i in indices:
        np.testing.assert_allclose(result.positions[i], original[i] + disp, atol=1e-10)
    # Non-moved atoms unchanged
    for i in range(len(ethane)):
        if i not in indices:
            np.testing.assert_allclose(result.positions[i], original[i], atol=1e-10)


def test_move_group_does_not_mutate(ethane):
    original = ethane.positions.copy()
    move_group(ethane, [0, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ethane.positions, original)


# ─────────────────────────────────────────────
# set_bond_length / get_bond_length
# ─────────────────────────────────────────────

def test_get_bond_length(ethane):
    # C-C bond should be ~1.54 Å
    d = get_bond_length(ethane, 0, 1)
    assert abs(d - 1.54) < 0.01


def test_set_bond_length(ethane):
    new_len = 1.60
    result = set_bond_length(ethane, 0, 1, new_len)
    measured = get_bond_length(result, 0, 1)
    assert abs(measured - new_len) < 1e-6


def test_set_bond_length_atom_i_fixed(ethane):
    result = set_bond_length(ethane, 0, 1, 2.0)
    # Atom 0 position must not change
    np.testing.assert_allclose(result.positions[0], ethane.positions[0], atol=1e-10)


def test_set_bond_length_does_not_mutate(ethane):
    original = ethane.positions.copy()
    set_bond_length(ethane, 0, 1, 2.0)
    np.testing.assert_allclose(ethane.positions, original)


# ─────────────────────────────────────────────
# set_bond_angle / get_bond_angle
# ─────────────────────────────────────────────

def test_get_bond_angle_water(water):
    # H-O-H angle in water ~104° (depends on the coordinate set used)
    angle = get_bond_angle(water, 1, 0, 2)
    assert 95.0 < angle < 115.0


def test_set_bond_angle(water):
    new_angle = 109.5
    result = set_bond_angle(water, 1, 0, 2, new_angle)
    measured = get_bond_angle(result, 1, 0, 2)
    assert abs(measured - new_angle) < 1e-4


def test_set_bond_angle_preserves_bond_length(water):
    """The O-H bond length (atom_j to atom_k) should be preserved when changing the angle."""
    original_oh = get_bond_length(water, 0, 2)
    result = set_bond_angle(water, 1, 0, 2, 120.0)
    new_oh = get_bond_length(result, 0, 2)
    assert abs(new_oh - original_oh) < 1e-5


def test_set_bond_angle_atom_j_fixed(water):
    result = set_bond_angle(water, 1, 0, 2, 120.0)
    np.testing.assert_allclose(result.positions[0], water.positions[0], atol=1e-10)


def test_set_bond_angle_does_not_mutate(water):
    original = water.positions.copy()
    set_bond_angle(water, 1, 0, 2, 120.0)
    np.testing.assert_allclose(water.positions, original)


# ─────────────────────────────────────────────
# set_dihedral / get_dihedral
# ─────────────────────────────────────────────

def test_get_dihedral_ethane(ethane):
    # H2-C0-C1-H5 dihedral
    d = get_dihedral(ethane, 2, 0, 1, 5)
    assert abs(d) <= 180.0


def test_set_dihedral(ethane):
    new_d = 60.0
    result = set_dihedral(ethane, 2, 0, 1, 5, new_d)
    measured = get_dihedral(result, 2, 0, 1, 5)
    assert abs(measured - new_d) < 1e-4


def test_set_dihedral_does_not_mutate(ethane):
    original = ethane.positions.copy()
    set_dihedral(ethane, 2, 0, 1, 5, 60.0)
    np.testing.assert_allclose(ethane.positions, original)


def test_set_dihedral_preserves_bond_length_jk(ethane):
    """The C-C bond (j-k) must not change when rotating the dihedral."""
    original_cc = get_bond_length(ethane, 0, 1)
    result = set_dihedral(ethane, 2, 0, 1, 5, 60.0)
    new_cc = get_bond_length(result, 0, 1)
    assert abs(new_cc - original_cc) < 1e-6


def test_set_dihedral_180(ethane):
    result = set_dihedral(ethane, 2, 0, 1, 5, 180.0)
    measured = get_dihedral(result, 2, 0, 1, 5)
    assert abs(abs(measured) - 180.0) < 1e-4


# ─────────────────────────────────────────────
# detect_fragment
# ─────────────────────────────────────────────

def test_detect_fragment_ethane(ethane):
    """For ethane C0-C1 bond, fragment starting from atom 1 (C1) should contain {1, 5, 6, 7}."""
    # Ethane atom numbering:
    # 0: C (left),  1: C (right)
    # 2,3,4: H on C0;  5,6,7: H on C1
    fragment = detect_fragment(ethane, 0, 1, 1)
    assert set(fragment) == {1, 5, 6, 7}, f"Expected {{1, 5, 6, 7}}, got {set(fragment)}"


def test_detect_fragment_ethane_other_side(ethane):
    """Fragment starting from atom 0 (C0 side) should contain {0, 2, 3, 4}."""
    fragment = detect_fragment(ethane, 0, 1, 0)
    assert set(fragment) == {0, 2, 3, 4}, f"Expected {{0, 2, 3, 4}}, got {set(fragment)}"


# ─────────────────────────────────────────────
# rotate_dihedral_fragment
# ─────────────────────────────────────────────

def test_rotate_dihedral_fragment_ethane(ethane):
    """Rotate fragment {1,5,6,7} around C-C bond (0,1) by 60°, verify dihedral changes and C-C preserved."""
    fragment = [1, 5, 6, 7]
    initial_dihedral = get_dihedral(ethane, 2, 0, 1, 5)
    initial_cc = get_bond_length(ethane, 0, 1)

    result = rotate_dihedral_fragment(ethane, 0, 1, 60.0, fragment)

    # C-C bond length preserved
    new_cc = get_bond_length(result, 0, 1)
    assert abs(new_cc - initial_cc) < 1e-6, f"C-C bond changed: {initial_cc} → {new_cc}"

    # Dihedral changed by ~60°
    new_dihedral = get_dihedral(result, 2, 0, 1, 5)
    delta = new_dihedral - initial_dihedral
    # Normalize to [-180, 180]
    delta = (delta + 180) % 360 - 180
    assert abs(abs(delta) - 60.0) < 1e-3, f"Expected dihedral change of ~60°, got {delta:.4f}°"

    # All atoms NOT in fragment are unchanged
    for i in range(len(ethane)):
        if i not in fragment:
            np.testing.assert_allclose(
                result.positions[i], ethane.positions[i], atol=1e-10,
                err_msg=f"Atom {i} (not in fragment) was moved"
            )

    # Fragment atoms moved together: relative inter-fragment distances preserved
    for a in fragment:
        for b in fragment:
            orig_dist = np.linalg.norm(ethane.positions[a] - ethane.positions[b])
            new_dist = np.linalg.norm(result.positions[a] - result.positions[b])
            assert abs(new_dist - orig_dist) < 1e-6, (
                f"Relative distance between fragment atoms {a}-{b} changed: {orig_dist:.6f} → {new_dist:.6f}"
            )


def test_rotate_dihedral_fragment_does_not_mutate(ethane):
    original = ethane.positions.copy()
    rotate_dihedral_fragment(ethane, 0, 1, 60.0, [1, 5, 6, 7])
    np.testing.assert_allclose(ethane.positions, original)


# ─────────────────────────────────────────────
# set_dihedral_fragment
# ─────────────────────────────────────────────

def test_set_dihedral_fragment(ethane):
    """Set dihedral to 60° using fragment rotation, verify within 0.1°."""
    fragment = detect_fragment(ethane, 0, 1, 1)
    target = 60.0
    result = set_dihedral_fragment(ethane, 2, 0, 1, 5, target, fragment)
    measured = get_dihedral(result, 2, 0, 1, 5)
    assert abs(measured - target) < 0.1, f"Expected {target}°, got {measured:.4f}°"


def test_set_dihedral_fragment_preserves_cc_bond(ethane):
    """C-C bond length must not change after fragment dihedral rotation."""
    fragment = detect_fragment(ethane, 0, 1, 1)
    original_cc = get_bond_length(ethane, 0, 1)
    result = set_dihedral_fragment(ethane, 2, 0, 1, 5, 60.0, fragment)
    new_cc = get_bond_length(result, 0, 1)
    assert abs(new_cc - original_cc) < 1e-6


def test_set_dihedral_fragment_does_not_mutate(ethane):
    original = ethane.positions.copy()
    fragment = detect_fragment(ethane, 0, 1, 1)
    set_dihedral_fragment(ethane, 2, 0, 1, 5, 60.0, fragment)
    np.testing.assert_allclose(ethane.positions, original)
