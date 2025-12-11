import random
from fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP(name="Demo Server (Roll Dice / Add Two Numbers)")

@mcp.tool
def roll_dice(n_dice: int = 1) -> list[int]:
    """Roll n_dice 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]
    
@mcp.tool
def add_numbers(a: float, b: float) -> int:
    """Add n numbers."""
    return a + b

if __name__ == "__main__":
    # for testing the server: uv run fastmcp dev 01_practice_mcp_server.py
    # for tunning the server: uv run fastmcp run 01_practice_mcp_server.py
    # for inegrating it to the claude desktop run: uv run fastmcp install claude-desktop 01_practice_mcp_server.py
    mcp.run()
    