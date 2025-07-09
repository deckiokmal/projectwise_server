from mcp_server.server import mcp

mcp = mcp

# run the server
if __name__ == "__main__":
    mcp.run(transport="sse")
