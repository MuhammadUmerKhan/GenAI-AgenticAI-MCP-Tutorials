from fastmcp import FastMCP

# Create a proxy to your remote FastMCP Cloud server
# FastMCP Cloud uses Streamable HTTP (default), so just use the /mcp URL
mcp = FastMCP.as_proxy(
    "https://handicapped-blush-jaguar.fastmcp.app/mcp",  # Standard FastMCP Cloud URL
    name="Expensse Tracker Proxy Server"
)

if __name__ == "__main__":
    # This runs via STDIO, which Claude Desktop can connect to
    mcp.run()
    
    # Run locally remote server on claude dsktop: uv run fastmcp install claude-desktop 02_proxy.py