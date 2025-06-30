import subprocess
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP()

@mcp.tool()
def find_files(root: str = ".", 
               maxdepth: int = None,
               mtime_gt_days: int = None, 
               size_gt_bytes: int = None, 
               name_contains: str = None) -> str:
    """
    Uses `find` command to emit a raw newline-separated list of file paths and their human-readable sizes.
    Each line is formatted as "SIZE\t/path/to/file" (e.g., "1.2G\\t/path/to/some/file").
    - maxdepth: descend at most levels of directories below the starting-points.
    - mtime_gt_days: only files modified more than this many days ago
    - size_gt_bytes: only files at least this many bytes
    - name_contains: only files whose basename contains this substring
    """
    cmd = ["find", root]
    if maxdepth is not None:
        cmd += ["-maxdepth", str(maxdepth)]
    if mtime_gt_days is not None:
        # find's "-mtime +N" means modified > N days ago
        cmd += ["-mtime", f"+{mtime_gt_days}"]
    if size_gt_bytes is not None:
        # "-size +N[c]" for bytes; GNU find uses 'c' for bytes, BSD find too
        cmd += ["-size", f"+{size_gt_bytes}c"]
    if name_contains:
        cmd += ["-iname", f"*{name_contains}*"]
    # only files
    cmd += ["-type", "f"]
    
    # Execute `du -h` on found files to get human-readable sizes
    cmd += ["-exec", "du", "-h", "{}", "+"]
    
    # safely build and run
    result = subprocess.check_output(cmd)
    return result.decode("utf-8", errors="ignore")

if __name__ == "__main__":
    # print(find_files("/Users/enzo/Documents/", mtime_gt_days=365, size_gt_bytes=100000000))
    mcp.run(transport="stdio")