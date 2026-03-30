"""
Integration tests for the MCP geometry server.

Each test:
1. Starts MCP server via stdio transport
2. Loads ethane.xyz fresh
3. Sends a prompt to the LLM and runs the agentic loop
4. Retrieves the molecule and verifies geometric properties numerically
"""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from geometry_tools import atoms_from_xyz, get_bond_length, get_bond_angle, get_dihedral
from mcp_utils import mcp_tool_to_openai

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

BASE_URL = os.environ["OPENAI_BASE_URL"]
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ["OPENAI_MODEL"]

ETHANE_XYZ_PATH = Path(__file__).parent.parent / "ethane.xyz"
MCP_SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_server.py")

SYSTEM_PROMPT = """\
You are a molecular geometry editor assistant. You have access to tools for editing molecular structures.
When the user asks you to make a geometric change, use the appropriate tool to perform the change.
Use exact atom indices as specified in the prompt.
Do not ask clarifying questions. Just perform the requested operation immediately.
"""

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def run_agentic_loop(
    session: ClientSession,
    client: AsyncOpenAI,
    openai_tools: list,
    tool_names: set,
    prompt: str,
) -> str:
    """Send a prompt to LLM, run the agentic loop, return final molecule XYZ."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for _ in range(20):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            logger.info("LLM finished. Final message: %s", msg.content)
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            if fn_name not in tool_names:
                tool_result_text = f"Error: unknown tool '{fn_name}'"
            else:
                logger.info("Tool call: %s(%s)", fn_name, fn_args)
                try:
                    result = await session.call_tool(fn_name, fn_args)
                    tool_result_text = result.content[0].text if result.content else "(no output)"
                except Exception as exc:
                    tool_result_text = f"Tool error: {exc}"
                logger.info("Tool result: %s", tool_result_text[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result_text})

    result = await session.call_tool("get_molecule", {})
    return result.content[0].text


@asynccontextmanager
async def mcp_test_session(server_params, xyz_string):
    """Start a fresh MCP server, initialize tools, load molecule, yield test helpers."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]
            tool_names = {t.name for t in tools_result.tools}
            client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
            await session.call_tool("load_molecule", {"xyz_string": xyz_string})
            yield session, openai_tools, tool_names, client


@pytest.fixture
def server_params():
    return StdioServerParameters(
        command="uv",
        args=["run", "python", MCP_SERVER_SCRIPT],
        env=None,
    )


@pytest.fixture
def ethane_xyz():
    return ETHANE_XYZ_PATH.read_text()


@pytest.mark.asyncio
async def test_move_single_atom(server_params, ethane_xyz):
    """Move atom 2 (hydrogen) by 0.5 Angstroms in the x direction and verify."""
    async with mcp_test_session(server_params, ethane_xyz) as (session, openai_tools, tool_names, client):
        initial_atoms = atoms_from_xyz((await session.call_tool("get_molecule", {})).content[0].text)
        initial_x = initial_atoms.positions[2][0]

        final_xyz = await run_agentic_loop(
            session, client, openai_tools, tool_names,
            "Move atom 2 (hydrogen) by 0.5 Angstroms in the x direction."
        )

        delta_x = atoms_from_xyz(final_xyz).positions[2][0] - initial_x
        logger.info("Atom 2 x delta: %.4f", delta_x)
        assert abs(delta_x - 0.5) < 0.05, f"Expected delta ~0.5, got {delta_x:.4f}"


@pytest.mark.asyncio
async def test_move_atom_group(server_params, ethane_xyz):
    """Move atoms 1, 5, 6, 7 (second carbon and its hydrogens) by 1.0 Å in y and verify."""
    async with mcp_test_session(server_params, ethane_xyz) as (session, openai_tools, tool_names, client):
        initial_atoms = atoms_from_xyz((await session.call_tool("get_molecule", {})).content[0].text)
        initial_ys = {idx: initial_atoms.positions[idx][1] for idx in [1, 5, 6, 7]}

        final_xyz = await run_agentic_loop(
            session, client, openai_tools, tool_names,
            "Move atoms 1, 5, 6, 7 (the second carbon and its hydrogens) by 1.0 Angstrom in the y direction."
        )

        final_atoms = atoms_from_xyz(final_xyz)
        for idx in [1, 5, 6, 7]:
            delta_y = final_atoms.positions[idx][1] - initial_ys[idx]
            logger.info("Atom %d y delta: %.4f", idx, delta_y)
            assert abs(delta_y - 1.0) < 0.05, f"Expected atom {idx} y delta ~1.0, got {delta_y:.4f}"


@pytest.mark.asyncio
async def test_change_bond_length(server_params, ethane_xyz):
    """Set C-C bond length (atoms 0 and 1) to 1.65 Å and verify."""
    async with mcp_test_session(server_params, ethane_xyz) as (session, openai_tools, tool_names, client):
        final_xyz = await run_agentic_loop(
            session, client, openai_tools, tool_names,
            "Set the C-C bond length (atoms 0 and 1) to 1.65 Angstroms."
        )

        bond_len = get_bond_length(atoms_from_xyz(final_xyz), 0, 1)
        logger.info("C-C bond length after: %.4f Å", bond_len)
        assert abs(bond_len - 1.65) < 0.01, f"Expected ~1.65 Å, got {bond_len:.4f} Å"


@pytest.mark.asyncio
async def test_change_bond_angle(server_params, ethane_xyz):
    """Change H-C-C angle (atoms 2, 0, 1) to 115 degrees and verify."""
    async with mcp_test_session(server_params, ethane_xyz) as (session, openai_tools, tool_names, client):
        final_xyz = await run_agentic_loop(
            session, client, openai_tools, tool_names,
            "Change the H-C-C angle (atoms 2, 0, 1) to 115 degrees."
        )

        angle = get_bond_angle(atoms_from_xyz(final_xyz), 2, 0, 1)
        logger.info("H-C-C angle after: %.4f°", angle)
        assert abs(angle - 115.0) < 0.5, f"Expected ~115°, got {angle:.4f}°"


@pytest.mark.asyncio
async def test_change_dihedral_angle(server_params, ethane_xyz):
    """Set dihedral H-C-C-H (atoms 2, 0, 1, 5) to 60 degrees and verify."""
    async with mcp_test_session(server_params, ethane_xyz) as (session, openai_tools, tool_names, client):
        final_xyz = await run_agentic_loop(
            session, client, openai_tools, tool_names,
            "Set the dihedral H-C-C-H (atoms 2, 0, 1, 5) to 60 degrees."
        )

        dihedral = get_dihedral(atoms_from_xyz(final_xyz), 2, 0, 1, 5)
        logger.info("H-C-C-H dihedral after: %.4f°", dihedral)
        assert abs(dihedral - 60.0) < 1.0, f"Expected ~60°, got {dihedral:.4f}°"


@pytest.mark.asyncio
async def test_change_dihedral_fragment(server_params, ethane_xyz):
    """Rotate the whole CH3 fragment via change_dihedral_angle_fragment,
    verify dihedral is correct AND internal fragment geometry is preserved."""
    async with mcp_test_session(server_params, ethane_xyz) as (session, openai_tools, tool_names, client):
        initial_atoms = atoms_from_xyz((await session.call_tool("get_molecule", {})).content[0].text)
        frag = [1, 5, 6, 7]
        initial_dists = {
            (a, b): float(np.linalg.norm(initial_atoms.positions[a] - initial_atoms.positions[b]))
            for a in frag for b in frag if a < b
        }

        final_xyz = await run_agentic_loop(
            session, client, openai_tools, tool_names,
            "Use the fragment dihedral tool to set the dihedral H-C-C-H "
            "(atoms 2, 0, 1, 5) to 180 degrees, rotating the entire CH3 group."
        )

        final_atoms = atoms_from_xyz(final_xyz)
        dihedral = get_dihedral(final_atoms, 2, 0, 1, 5)
        assert abs(abs(dihedral) - 180.0) < 1.0, f"Expected ~180°, got {dihedral:.4f}°"

        for (a, b), orig_dist in initial_dists.items():
            new_dist = float(np.linalg.norm(final_atoms.positions[a] - final_atoms.positions[b]))
            assert abs(new_dist - orig_dist) < 0.01, (
                f"Fragment distance {a}-{b} changed: {orig_dist:.4f} → {new_dist:.4f}"
            )
