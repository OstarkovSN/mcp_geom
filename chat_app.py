"""
Chat application that connects to the MCP geometry server and uses OpenAI for tool-augmented conversation.
Usage:
    uv run python chat_app.py [molecule.xyz]

If molecule.xyz is given, it is loaded automatically.
Reads OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL from .env
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from mcp_utils import mcp_tool_to_openai

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_URL = os.environ["OPENAI_BASE_URL"]
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ["OPENAI_MODEL"]

SYSTEM_PROMPT = """\
You are a molecular geometry editor assistant. You have access to tools for editing molecular structures.
When the user asks you to make a geometric change (move atoms, change bond lengths/angles/dihedrals),
use the appropriate tool to perform the change and report the result clearly.
Always call get_atom_info first if you need to identify atom indices.
Report numeric results (before/after values) clearly to the user.
"""


async def run_chat(initial_xyz_path: str | None = None):
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]
            tool_names = {t.name for t in tools_result.tools}

            client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

            # Load initial molecule if provided
            if initial_xyz_path:
                xyz_content = Path(initial_xyz_path).read_text()
                result = await session.call_tool("load_molecule", {"xyz_string": xyz_content})
                print(f"[MCP] {result.content[0].text}")

            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

            print("\nMolecular Geometry Editor Chat")
            print("Type your request, or 'quit' to exit.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    break

                if user_input.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break

                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                # Agentic loop: keep calling until no more tool calls
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
                        # Final text response
                        print(f"\nAssistant: {msg.content}\n")
                        break

                    # Execute all tool calls
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)

                        if fn_name not in tool_names:
                            tool_result_text = f"Error: unknown tool '{fn_name}'"
                        else:
                            print(f"  [tool] {fn_name}({json.dumps(fn_args, separators=(',', ':'))})")
                            try:
                                result = await session.call_tool(fn_name, fn_args)
                                tool_result_text = result.content[0].text if result.content else "(no output)"
                            except Exception as exc:
                                tool_result_text = f"Tool error: {exc}"

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result_text,
                        })


def main():
    xyz_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_chat(xyz_path))


if __name__ == "__main__":
    main()
