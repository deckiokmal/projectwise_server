import logging
import sys
from pathlib import Path

# Buat folder logs jika belum ada
log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Tentukan log file berdasarkan lokasi caller
caller_path = Path(sys.argv[0]).resolve()
if "mcp_server" in str(caller_path):
    log_file = log_dir / "mcp_server.log"
elif "mcp_client" in str(caller_path):
    log_file = log_dir / "mcp_client.log"
else:
    log_file = log_dir / "mcp_generic.log"

logger = logging.getLogger("MCPLogger")
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)

logger.debug(f"Logger initialized from {caller_path}, writing to {log_file}")
