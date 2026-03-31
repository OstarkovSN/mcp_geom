# mcp-geom

An MCP (Model Context Protocol) server for molecular geometry editing, powered by [ASE](https://wiki.fysik.dtu.dk/ase/). Includes a CLI chat app that lets an LLM edit molecular structures via natural-language prompts.

## Features

- **Move atoms** — single atom or group by a displacement vector
- **Bond lengths** — measure and set distances between atoms
- **Bond angles** — measure and set angles, with optional rigid-fragment rotation
- **Dihedral angles** — measure and set torsion angles, with optional rigid-fragment rotation
- **Whole-molecule transforms** — center at origin, rotate around an axis
- **Save/load** — load from XYZ string, save to `.xyz` file

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11+
- An OpenAI-compatible API endpoint

## Setup

```bash
git clone <repo>
cd mcp-geom
uv sync
```

Create a `.env` file:

```ini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

## Usage

### Interactive chat

```bash
uv run python chat_app.py ethane.xyz
```

Example prompts:
```
Set the C-C bond length to 1.65 Angstroms
Change the H-C-C angle (atoms 2, 0, 1) to 115 degrees
Rotate the dihedral H-C-C-H (atoms 2, 0, 1, 5) to 60 degrees
Rotate the entire CH3 group so the dihedral is 180 degrees
Get atom info
Save the molecule to output.xyz
```

Type `quit` to exit.

### Visualize results

```bash
uv run ase gui output.xyz
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `load_molecule` | Load a molecule from an XYZ string |
| `get_molecule` | Return current molecule as XYZ |
| `get_atom_info` | Table of atom indices, symbols, and coordinates |
| `move_single_atom` | Move one atom by (dx, dy, dz) |
| `move_atom_group` | Move a list of atoms by (dx, dy, dz) |
| `measure_bond_length` | Distance between two atoms |
| `change_bond_length` | Set distance between two atoms (moves atom_j) |
| `measure_bond_angle` | Angle i-j-k in degrees |
| `change_bond_angle` | Set angle i-j-k (moves atom_k only) |
| `change_bond_angle_fragment` | Set angle i-j-k, rotating entire fragment on atom_k side |
| `measure_dihedral_angle` | Dihedral i-j-k-l in degrees |
| `change_dihedral_angle` | Set dihedral i-j-k-l (moves atom_l only) |
| `change_dihedral_angle_fragment` | Set dihedral i-j-k-l, rotating entire fragment on atom_l side |
| `get_molecule_center_of_mass` | Return center of mass coordinates |
| `center_molecule_at_origin` | Translate molecule so COM is at origin |
| `rotate_whole_molecule` | Rotate entire molecule around an axis through COM |
| `save_molecule` | Save molecule to a `.xyz` file |

### Fragment tools

The `_fragment` variants (`change_bond_angle_fragment`, `change_dihedral_angle_fragment`) auto-detect all atoms on the moving side of a bond via BFS and rotate them rigidly as a group. This preserves internal geometry — useful for rotating substituents without distorting them.

## Running the MCP server standalone

```bash
uv run python mcp_server.py
```

Communicates via stdio (standard MCP transport). Connect with any MCP client.

## Tests

```bash
# Unit tests (no API required)
uv run pytest tests/test_geometry_tools.py -v

# End-to-end LLM integration tests (requires .env)
uv run pytest tests/test_success_criteria.py -v

# All tests
uv run pytest
```

The integration tests start a fresh MCP server subprocess for each test, load `ethane.xyz`, send a natural-language prompt to the LLM, and verify the resulting geometry numerically.

## Project structure

```
mcp_server.py          # FastMCP server with 17 tools
geometry_tools.py      # Pure geometry functions (ASE Atoms → ASE Atoms)
chat_app.py            # CLI chat app (OpenAI + MCP)
mcp_utils.py           # Shared MCP→OpenAI tool schema conversion
ethane.xyz             # Sample molecule (C2H6)
tests/
  test_geometry_tools.py    # 43 unit tests
  test_success_criteria.py  # 7 LLM integration tests
```

## Architecture

```
chat_app.py  ──stdio──►  mcp_server.py
     │                        │
     │                   geometry_tools.py
     │                   (in-memory Atoms)
     │
  OpenAI API
```

The MCP server holds the molecule in memory between tool calls. The chat app bridges LLM function calls to MCP tool calls — the XYZ coordinates are not re-sent with every message.
