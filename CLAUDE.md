Your goal is to design, implement and test an MCP server for molecular geometry editing. It should have a bunch of tools,
including but not limited to:
+ Moving the atoms/groups
+ Lengthening/shortening the  bonds
+ Changing the bond angles
+ Rotating around the bond to change  the dihedral angle.
Use ASE (and rdkit if needed).
After implementing an MCP with all the tools and automatic unit tests (via pytest), you should test it.
Write a simple overhead chat with LLM app (no GUI if you won't feel tht you need it) that uses the mcp and tools.
Use .env file in this repo for OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL
Don't change any of those.
Criteria of success:
You feed the model a molecular geometry (as plain .xyz), and for each of the tools, ask a prompt that will use this tool. Model uses the tool
successfully (!) and the result is compliant with the prompt.
Don't ask for permissions or decisions, the user is sleeping.
