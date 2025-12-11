import random, json
from fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP(name="FastMCP Server")

@mcp.tool
def generate_random_numbers(n: int) -> list[int]:
    """
    Generate n random numbers.

    Args:
        n (int): The number of random numbers to generate.

    Returns:
        list[int]: A list of n random numbers.
    """
    # Generate n random numbers between 1 and 100
    """Generate n random numbers."""
    return [random.randint(1, 100) for _ in range(n)]
    
@mcp.tool
def add_numbers(a: float, b: float) -> int:
    """
    Add two numbers.
    
    Args:
        a (float): The first number to add.
        b (float): The second number to add.
    
    Returns:
        int: The sum of a and b.
    """
    return int(a + b)

# Resource: Server Information
@mcp.resource("info://server")
def server_info() -> str:
    """Get information about the server."""
    info = {
        "name": "Simple Demo Calculator Server",
        "description": "A simple calculator server that can add and generate random numbers.",
        "tools": ["add_numbers", "generate_random_numbers"],
        "version": "1.0.0",
        "author": "Umer Khan"
    }
    return json.dumps(info, indent=2)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
    # for running the server: fastmcp run 01_practice_mcp_server.py --transport http --host 0.0.0.0 --port 8000 or uv run 01_practice_mcp_server.py
    # In debugging mode: uv run fastmcp dev 01_practice_mcp_server.py